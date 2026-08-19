import os
import sys
import json
import base64
import wave
import io
import math
import textwrap
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).parent.resolve()))
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import textgrid functions
try:
    from textgrid_transfrom import parse_textgrid_intervals, extract_text_from_textgrid
except Exception:
    try:
        from audio.textgrid_transfrom import parse_textgrid_intervals, extract_text_from_textgrid
    except Exception:
        try:
            from models.sound.textgrid_transfrom import parse_textgrid_intervals, extract_text_from_textgrid
        except Exception:
            parse_textgrid_intervals = None
            extract_text_from_textgrid = None

def apply_common_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=Inter:wght@400;600&display=swap');
        
        /* Force Full Width (100%) Layout with minimal padding */
        .block-container {
            max-width: 100% !important;
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        .main-title {
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(90deg, #4A90E2, #8E2DE2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.8rem;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-family: 'Inter', sans-serif;
            color: #7f8c8d;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 8px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-weight: 600;
            font-size: 16px;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 2px solid #4A90E2 !important;
            color: #4A90E2 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Helper function to slice audio segment in memory and return bytes
def slice_audio_bytes(audio_path, start_sec, end_sec):
    if not audio_path or not Path(audio_path).exists():
        return None
    
    audio_path = Path(audio_path)
    try:
        start_sec = max(0.0, float(start_sec))
        end_sec = max(start_sec, float(end_sec))
    except (ValueError, TypeError):
        return None

    if end_sec <= start_sec:
        return None

    if audio_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                framerate = wf.getframerate()
                nchannels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                nframes = wf.getnframes()
                
                start_frame = max(0, int(start_sec * framerate))
                end_frame = min(nframes, int(end_sec * framerate))
                num_frames = end_frame - start_frame
                if num_frames <= 0:
                    return None
                
                wf.setpos(start_frame)
                frames = wf.readframes(num_frames)
                
                out_buffer = io.BytesIO()
                with wave.open(out_buffer, 'wb') as out_wf:
                    out_wf.setnchannels(nchannels)
                    out_wf.setsampwidth(sampwidth)
                    out_wf.setframerate(framerate)
                    out_wf.writeframes(frames)
                
                return out_buffer.getvalue()
        except Exception:
            pass

    try:
        import subprocess
        dur = max(0.05, end_sec - start_sec)
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_sec), "-i", str(audio_path),
            "-t", str(dur), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-f", "wav", "pipe:1"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        return res.stdout
    except Exception:
        return None

def get_audio_segment_html(audio_path, start_sec, end_sec):
    audio_bytes = slice_audio_bytes(audio_path, start_sec, end_sec)
    if not audio_bytes:
        return "<span style='color:#888; font-size:0.8rem;'>재생 불가</span>"
    
    b64_str = base64.b64encode(audio_bytes).decode("utf-8")
    return f'<audio controls src="data:audio/wav;base64,{b64_str}" style="height: 32px; width: 220px; vertical-align: middle;"></audio>'

def get_audio_duration(audio_path):
    if not audio_path or not Path(audio_path).exists():
        return 0.0
    try:
        with wave.open(str(audio_path), 'rb') as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        try:
            import subprocess
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            return float(res.stdout.strip())
        except Exception:
            return 0.0

def compute_non_speech_intervals(speech_intervals, total_duration, min_gap=0.1):
    if not speech_intervals:
        if total_duration > 0:
            return [{"start": 0.0, "end": round(float(total_duration), 2)}]
        return []
    
    # Extract and filter valid speech segments
    valid_segs = []
    for s in speech_intervals:
        try:
            st_val = float(s.get("start", 0))
            en_val = float(s.get("end", 0))
            if en_val > st_val:
                valid_segs.append({"start": st_val, "end": en_val})
        except (ValueError, TypeError):
            continue

    if not valid_segs:
        if total_duration > 0:
            return [{"start": 0.0, "end": round(float(total_duration), 2)}]
        return []

    # Sort segments by start time
    sorted_segs = sorted(valid_segs, key=lambda x: x["start"])

    # Merge overlapping and contiguous speech segments
    merged_speech = []
    for seg in sorted_segs:
        if not merged_speech:
            merged_speech.append(dict(seg))
        else:
            last = merged_speech[-1]
            if seg["start"] <= last["end"]:
                last["end"] = max(last["end"], seg["end"])
            else:
                merged_speech.append(dict(seg))

    non_speech = []

    # Non-speech gap before the first speech segment
    first_start = merged_speech[0]["start"]
    if first_start - 0.0 >= min_gap:
        non_speech.append({
            "start": 0.0,
            "end": round(first_start, 2)
        })

    # Non-speech gaps between consecutive merged speech segments
    for i in range(len(merged_speech) - 1):
        prev_end = merged_speech[i]["end"]
        next_start = merged_speech[i + 1]["start"]
        if next_start - prev_end >= min_gap:
            non_speech.append({
                "start": round(prev_end, 2),
                "end": round(next_start, 2)
            })

    # Non-speech gap after the last speech segment
    last_end = merged_speech[-1]["end"]
    if total_duration > 0 and (float(total_duration) - last_end >= min_gap):
        non_speech.append({
            "start": round(last_end, 2),
            "end": round(float(total_duration), 2)
        })

    return non_speech

def render_paginated_segments(items_to_render, key_prefix, audio_source_path, items_per_page=20):
    if not items_to_render:
        st.info("표시할 구간 데이터가 없습니다.")
        return

    total_items = len(items_to_render)
    total_pages = max(1, math.ceil(total_items / items_per_page))

    if total_pages > 1:
        c1, c2, c3 = st.columns([1, 3, 1])
        with c2:
            page = st.number_input(
                f"📄 페이지 이동 (총 {total_items}개 구간 / {total_pages} 페이지)",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
                key=f"{key_prefix}_page"
            )
    else:
        page = 1

    start_idx = (page - 1) * items_per_page
    end_idx = min(total_items, page * items_per_page)
    page_items = items_to_render[start_idx:end_idx]

    st.caption(f"📌 {start_idx + 1} ~ {end_idx}번째 구간 표시 중 (전체 {total_items}개 구간)")

    import textwrap

    for idx, seg in enumerate(page_items, start=start_idx + 1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        dur = end - start
        seg_type = seg.get("type", "speech")
        audio_player_html = get_audio_segment_html(audio_source_path, start, end)

        if seg_type == "speech":
            spk = seg.get("speaker", None)
            color = seg.get("color", "#4A90E2")
            if spk:
                badge_html = f'<span style="background-color: {color}; color: white; border-radius: 12px; padding: 3px 12px; font-weight: bold; font-size: 0.85rem;">{spk}</span>'
            else:
                badge_html = '<span style="background-color: #4A90E2; color: white; border-radius: 12px; padding: 2px 10px; font-weight: bold; font-size: 0.85rem;">🗣️ 음성</span>'

            st.markdown(textwrap.dedent(f"""
            <div style="background-color: #f8f9fa; border-left: 5px solid {color}; border-radius: 8px; padding: 10px 16px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    {badge_html}
                    <span style="font-weight: bold; color: #212529;">구간 #{idx}</span> : 
                    <code style="color: {color}; font-weight: 600;">{start:.2f}s ~ {end:.2f}s</code> 
                    <span style="color: #868e96; font-size: 0.85rem;">(지속 시간: {dur:.2f}초)</span>
                </div>
                <div>
                    {audio_player_html}
                </div>
            </div>
            """).strip(), unsafe_allow_html=True)
        else:
            st.markdown(textwrap.dedent(f"""
            <div style="background-color: #f1f3f5; border-left: 5px solid #868e96; border-radius: 8px; padding: 10px 16px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="background-color: #868e96; color: white; border-radius: 12px; padding: 2px 10px; font-weight: bold; font-size: 0.85rem;">🔇 비음성</span>
                    <span style="font-weight: bold; color: #495057;">구간 #{idx}</span> : 
                    <code style="color: #495057; font-weight: 600;">{start:.2f}s ~ {end:.2f}s</code> 
                    <span style="color: #868e96; font-size: 0.85rem;">(지속 시간: {dur:.2f}초)</span>
                </div>
                <div>
                    {audio_player_html}
                </div>
            </div>
            """).strip(), unsafe_allow_html=True)

def draw_yolo_boxes(image_path, yolo_data):
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        COLORS = ["#FF3B30", "#007AFF", "#34C759", "#FF9500", "#5856D6", "#AF52DE", "#FF2D55"]
        
        img_info = yolo_data["images"][0]
        results = img_info.get("results", [])
        
        for i, det in enumerate(results):
            box = det["box"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color = COLORS[i % len(COLORS)]
            
            line_width = max(3, int(min(image.size) * 0.004))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            label = f"{det['name']} {det['confidence'] * 100:.1f}%"
            font_size = max(14, int(min(image.size) * 0.018))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()
                
            try:
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                padding = 4
                text_bbox = [text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding]
                draw.rectangle(text_bbox, fill=color)
            except AttributeError:
                draw.rectangle([x1, y1 - font_size - 4, x1 + len(label)*8, y1], fill=color)
                
            draw.text((x1, y1), label, fill="white", font=font)
            
        return image
    except Exception as e:
        st.error(f"Error drawing bounding boxes: {e}")
        return None

def render_yolo_dashboard(yolo_data):
    COLORS = ["#FF3B30", "#007AFF", "#34C759", "#FF9500", "#5856D6", "#AF52DE", "#FF2D55"]
    
    img_info = yolo_data["images"][0]
    speed = img_info.get("speed", {})
    results = img_info.get("results", [])
    shape = img_info.get("shape", [0, 0])
    
    preprocess = speed.get("preprocess", 0)
    inference = speed.get("inference", 0)
    postprocess = speed.get("postprocess", 0)
    
    metadata = yolo_data.get("metadata", {})
    function_time_call_ms = metadata.get("functionTimeCall", 0) * 1000
    total_speed = preprocess + inference + postprocess
    network_time = max(0.0, function_time_call_ms - total_speed)
    
    col_header_left, col_header_right = st.columns([4, 1])
    with col_header_left:
        st.markdown("<h3 style='margin: 0;'>Results</h3>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: #7f8c8d; font-size: 0.95rem;'>{len(results)} detections found · {shape[1]}x{shape[0]}px</span>", unsafe_allow_html=True)
    with col_header_right:
        st.markdown(
            f"<div style='float: right; background-color: #f1f3f5; border-radius: 12px; padding: 4px 12px; font-weight: bold; font-size: 0.9rem; color: #495057;'>{round(total_speed)} ms</div>", 
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    card_cols = st.columns(4)
    labels = ["Preprocess", "Inference", "Postprocess", "Network"]
    values = [preprocess, inference, postprocess, network_time]
    
    for i, col in enumerate(card_cols):
        with col:
            st.markdown(f"""
            <div style='background-color: #f8f9fa; border-radius: 8px; padding: 12px; text-align: center; border: 1px solid #e9ecef;'>
                <div style='font-size: 1.15rem; font-weight: 700; color: #212529;'>{values[i]:.1f} ms</div>
                <div style='font-size: 0.75rem; color: #868e96;'>{labels[i]}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    st.markdown("<h4 style='margin-bottom: 12px;'>Detections</h4>", unsafe_allow_html=True)
    if results:
        for i, det in enumerate(results):
            name = det.get("name", "Unknown")
            confidence = det.get("confidence", 0)
            color = COLORS[i % len(COLORS)]
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; background-color: #f8f9fa; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {color};">
                <div style="display: flex; align-items: center;">
                    <span style="height: 10px; width: 10px; background-color: {color}; border-radius: 50%; display: inline-block; margin-right: 12px;"></span>
                    <span style="font-weight: 600; color: #212529; font-size: 0.95rem;">{name}</span>
                </div>
                <span style="font-weight: 600; color: #495057; font-size: 0.95rem;">{confidence * 100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No objects detected.")
        
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    version = metadata.get("version", {})
    ultralytics_ver = version.get("ultralytics", "8.4.103")
    torch_ver = version.get("torch", "2.13.0+cpu")
    st.markdown(f"<div style='font-size: 0.75rem; color: #adb5bd;'>Ultralytics {ultralytics_ver} · PyTorch {torch_ver}</div>", unsafe_allow_html=True)

def get_mime_type(filename, category):
    ext = Path(filename).suffix.lower()
    if category == "doc" and ext == ".pdf":
        return "application/pdf"
    elif category == "image":
        return f"image/{ext[1:]}"
    elif category == "video":
        return f"video/{ext[1:]}"
    elif category == "sound":
        return f"audio/{ext[1:]}"
    return "application/octet-stream"

def get_model_outputs(model_dir, stem):
    json_path = None
    md_path = None
    if model_dir and model_dir.exists():
        for f in os.listdir(model_dir):
            if f.startswith(stem):
                f_path = model_dir / f
                if f.endswith(".json"):
                    json_path = f_path
                elif f.endswith((".md", ".txt")):
                    md_path = f_path
    return md_path, json_path
