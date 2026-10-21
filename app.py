"""Groq-powered multimodal video summarizer for Gradio/Render.

Pipeline: uploaded video -> audio extraction -> Groq Whisper transcription ->
representative frame sampling -> Groq vision analysis -> combined summary.
"""

import base64
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import imageio_ffmpeg
from groq import Groq

# Models currently documented by Groq for speech, vision, and text generation.
TRANSCRIPTION_MODEL = "whisper-large-v3-turbo"
VISION_MODEL = "qwen/qwen3.8-27b"
SUMMARY_MODEL = "openai/gpt-oss-120b"

AUDIO_CHUNK_SECONDS = 60
FRAME_INTERVAL_SECONDS = 30
DEFAULT_MAX_FRAMES = 9
HARD_MAX_FRAMES = 15
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_VIDEO_SECONDS = 15 * 60  # Keep free-hosted requests manageable.


def _get_video_info(video_path: str) -> dict[str, float | int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("The uploaded file could not be opened as a video. Try an MP4 file.")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError("Could not read the video's frame rate or duration.")

    duration = frame_count / fps
    if duration <= 0:
        raise ValueError("The uploaded video has no readable duration.")
    if duration > MAX_VIDEO_SECONDS:
        raise ValueError("For this free deployment, use a video that is 15 minutes or shorter.")

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }


def _run_ffmpeg_audio(video_path: str, audio_dir: Path) -> list[Path]:
    """Extract mono 16 kHz WAV audio in 60-second chunks."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output_pattern = str(audio_dir / "chunk_%03d.wav")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "segment",
        "-segment_time", str(AUDIO_CHUNK_SECONDS),
        "-reset_timestamps", "1",
        output_pattern,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    files = sorted(audio_dir.glob("chunk_*.wav"))
    # A video without an audio stream may make ffmpeg return non-zero; visual
    # summarization can still proceed in that case.
    if result.returncode != 0 and not files:
        return []
    return files


def _transcribe_audio(client: Groq, audio_files: list[Path], duration: float, progress: Any) -> str:
    segments: list[str] = []
    for index, audio_path in enumerate(audio_files):
        start_seconds = index * AUDIO_CHUNK_SECONDS
        end_seconds = min(start_seconds + AUDIO_CHUNK_SECONDS, duration)
        progress(0.15 + 0.30 * ((index + 1) / max(1, len(audio_files))),
                 desc=f"Transcribing audio chunk {index + 1}/{len(audio_files)}")
        with audio_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model=TRANSCRIPTION_MODEL,
                response_format="json",
                temperature=0.0,
            )
        text = (getattr(result, "text", "") or "").strip()
        if text:
            segments.append(f"[{_format_time(start_seconds)} - {_format_time(end_seconds)}]\n{text}")
    return "\n\n".join(segments)


def _format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _extract_frames(video_path: str, duration: float, max_frames: int) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not reopen the video to sample frames.")

    # Sample at roughly 30-second intervals; for longer videos spread the
    # requested number of frames across the full duration.
    count = min(max(1, int(max_frames)), HARD_MAX_FRAMES,
                max(1, math.ceil(duration / FRAME_INTERVAL_SECONDS)))
    if count == 1:
        timestamps = [0.0]
    else:
        timestamps = [i * max(0.0, duration - 0.5) / (count - 1) for i in range(count)]

    frames: list[dict[str, Any]] = []
    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()
        if not success or frame is None:
            continue

        height, width = frame.shape[:2]
        if width > 768:
            scale = 768 / width
            frame = cv2.resize(frame, (768, max(1, int(height * scale))))

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if not ok:
            continue
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        frames.append({"timestamp": timestamp, "image_b64": image_b64})

    cap.release()
    if not frames:
        raise ValueError("Could not extract any readable video frames.")
    return frames


def _analyse_frames(client: Groq, frames: list[dict[str, Any]], progress: Any) -> str:
    observations: list[str] = []
    batches = [frames[i:i + 3] for i in range(0, len(frames), 3)]  # Groq Qwen accepts max 3 images/request.

    for batch_index, batch in enumerate(batches):
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "Analyze these representative video frames in chronological order. "
                "For each frame, describe the visible scene, slides or screen text that is legible, "
                "objects/actions, and relevant transitions. Do not guess unseen facts. "
                "Each frame is labelled with its timestamp below."
            ),
        }]
        for frame in batch:
            content.append({
                "type": "text",
                "text": f"Frame timestamp: {_format_time(frame['timestamp'])}",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame['image_b64']}"},
            })

        progress(0.48 + 0.23 * ((batch_index + 1) / max(1, len(batches))),
                 desc=f"Analyzing visual frames {batch_index + 1}/{len(batches)}")
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_completion_tokens=1800,
        )
        observations.append(response.choices[0].message.content or "No visual observations returned.")
    return "\n\n".join(observations)


def _make_summary(client: Groq, transcript: str, visual_analysis: str, duration: float, progress: Any) -> str:
    progress(0.76, desc="Combining speech and visual evidence into the summary")
    # Bound prompt size for long recordings and free-hosted deployment.
    transcript_for_prompt = transcript[:65000] if transcript else "No speech transcript available; the video may be silent."
    visuals_for_prompt = visual_analysis[:25000] if visual_analysis else "No visual analysis available."
    prompt = f"""Create an accurate multimodal summary of a video lasting {_format_time(duration)}.
Combine the timestamped speech transcript with the observations from representative video frames.
Do not invent information. If speech and visual evidence disagree, state only what is supported.
Return these headings:
# Video title
# Overall summary (150-250 words)
# Key points
# Timestamped chapters (use approximate timestamps)
# Important visual moments
# Key takeaways (5-10 items)
# Action items (only when truly present)
# One-sentence summary

TIMESTAMPED SPEECH TRANSCRIPT:
{transcript_for_prompt}

VISUAL FRAME OBSERVATIONS:
{visuals_for_prompt}
"""
    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[
            {"role": "system", "content": "You are an accurate video summarization assistant. Be concise, grounded, and transparent about missing evidence."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=3000,
    )
    progress(0.98, desc="Finishing the report")
    return response.choices[0].message.content or "The model returned an empty summary."


def summarize_video(video_upload: Any, max_frames: int, progress=gr.Progress()) -> tuple[str, str, str, str]:
    if not video_upload:
        return "", "", "", "Please upload a video file first."

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "", "", "", "**Missing API key:** Add `GROQ_API_KEY` in Render → Environment, then redeploy."

    video_path = video_upload if isinstance(video_upload, str) else getattr(video_upload, "name", None)
    if not video_path or not Path(video_path).is_file():
        return "", "", "", "The uploaded video path could not be found. Please upload it again."

    if Path(video_path).stat().st_size > MAX_UPLOAD_BYTES:
        return "", "", "", "The video is over 100 MB. Please upload a smaller video."

    try:
        progress(0.03, desc="Reading video details")
        info = _get_video_info(video_path)
        duration = float(info["duration"])
        client = Groq(api_key=api_key)

        with tempfile.TemporaryDirectory(prefix="video_summary_") as temp_dir:
            work_dir = Path(temp_dir)
            audio_dir = work_dir / "audio_chunks"
            audio_dir.mkdir(parents=True, exist_ok=True)

            progress(0.10, desc="Extracting audio")
            audio_files = _run_ffmpeg_audio(video_path, audio_dir)
            if audio_files:
                transcript = _transcribe_audio(client, audio_files, duration, progress)
            else:
                transcript = "No audio track was detected; this summary will rely on the visual content."

            progress(0.46, desc="Sampling representative frames")
            frames = _extract_frames(video_path, duration, int(max_frames))
            visual_analysis = _analyse_frames(client, frames, progress)
            summary = _make_summary(client, transcript, visual_analysis, duration, progress)

        status = (
            f"**Completed successfully.** Duration: {_format_time(duration)} · "
            f"Audio chunks: {len(audio_files)} · Frames analyzed: {len(frames)}"
        )
        if not audio_files:
            status += " · No audio was available, so the summary is based on visuals."
        return summary, transcript, visual_analysis, status

    except Exception as exc:
        # Avoid printing secrets and give an actionable error in the UI.
        error_text = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
        return "", "", "", f"**Summarization failed:** `{error_text[:700]}`\n\nCheck the Render logs and confirm the Groq key and model availability."


with gr.Blocks(title="Groq Multimodal Video Summarizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🎬 Multimodal Video Summarizer\n"
        "Upload a short video. The app transcribes speech with Groq Whisper, analyzes representative frames with a vision model, "
        "and combines both modalities into a timestamped summary. **Recommended: videos up to 10 minutes; hard limit: 15 minutes / 100 MB.**"
    )

    with gr.Row():
        video_input = gr.File(
            label="Upload video (MP4, MOV, MKV, or WEBM)",
            file_types=[".mp4", ".mov", ".mkv", ".webm", ".avi"],
            type="filepath",
        )
        max_frames_input = gr.Slider(
            minimum=3, maximum=HARD_MAX_FRAMES, value=DEFAULT_MAX_FRAMES, step=1,
            label="Maximum frames to analyze",
        )

    run_button = gr.Button("Summarize video", variant="primary")
    status_output = gr.Markdown("Upload a video and click **Summarize video**.")
    summary_output = gr.Markdown(label="Final video summary")

    with gr.Accordion("Timestamped speech transcript", open=False):
        transcript_output = gr.Textbox(label="Transcript", lines=12, max_lines=30)
    with gr.Accordion("Visual frame analysis", open=False):
        visual_output = gr.Textbox(label="Visual observations", lines=12, max_lines=30)

    run_button.click(
        fn=summarize_video,
        inputs=[video_input, max_frames_input],
        outputs=[summary_output, transcript_output, visual_output, status_output],
    )

    gr.Markdown(
        "---\n**Deployment note:** Configure `GROQ_API_KEY` as an environment variable in Render. "
        "Do not put your API key in this file or in GitHub."
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.queue(max_size=2).launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
        max_file_size="100mb",
    )
