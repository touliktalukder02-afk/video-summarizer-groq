# Q4 — Multimodal Video Summarization with Groq + Gradio

This app follows the provided assignment's workflow:
1. Upload a video.
2. Extract and split the audio into 60-second WAV chunks with FFmpeg.
3. Transcribe audio using Groq Whisper (`whisper-large-v3-turbo`).
4. Sample representative frames from the video.
5. Analyze batches of up to three images using Groq's vision model (`qwen/qwen3.8-27b`).
6. Fuse the transcript and visual observations into a timestamped summary using `openai/gpt-oss-120b` via Groq.

## Deploy on Render (recommended)

1. Create a **new public GitHub repository**, e.g. `video-summarizer-groq`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. In Render, select **New + → Web Service** and connect the repository.
4. Choose Python 3.
5. Build command: `pip install -r requirements.txt`
6. Start command: `python app.py`
7. Choose the Free instance if available.
8. In Environment Variables, add `GROQ_API_KEY` with your Groq secret as the value. Never commit the key to GitHub.
9. Click **Deploy Web Service** and wait for `Live`.
10. Open the Render service URL, upload a short video, and click **Summarize video**.

## Notes

- The implementation is based on the assignment's original pipeline, but updates the client to use Groq and wraps it in a Gradio UI.
- Groq vision model requests are limited to three images per request in the documented model, so frames are sent in batches.
- The service enforces a 100 MB file limit and 15-minute video limit to keep free hosting manageable.
- A short video with speech is best for the assignment demonstration.
- If the selected Groq vision model is unavailable to your account, check the current Groq model catalogue and replace `VISION_MODEL` with a currently enabled vision-capable model.
