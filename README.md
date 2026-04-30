# YouTube Video Q&A (Streamlit + RAG)

A production-ready Streamlit app that answers questions about a YouTube video using transcript-based RAG (Retrieval-Augmented Generation), timestamped evidence, and conversational memory.

## Highlights

- Accepts any valid YouTube URL
- Fetches metadata + transcript automatically
- Uses hybrid retrieval (`FAISS` semantic + `BM25` keyword)
- Returns answers with transcript timestamps like `[12:35]`
- Supports follow-up questions with chat memory
- Generates automatic chapter suggestions
- Adaptive answer style:
  - Short answers for direct questions
  - Medium answers for explainers
  - Detailed answers for broad/multi-part queries
  - Markdown formatting (bullets, numbered lists, bold/italic) when useful

## Tech Stack

- UI: `Streamlit`
- LLM: `Groq` (`llama-3.1-8b-instant`)
- Embeddings: `Google Gemini Embeddings` (`models/gemini-embedding-001`)
- Vector DB: `FAISS`
- Keyword Retrieval: `BM25`
- Transcript Sources:
  1. `youtube-transcript-api` (fastest path)
  2. Groq Whisper API fallback
  3. Local Whisper fallback

## Project Structure

- `app.py` - Streamlit UI and chat workflow
- `rag_pipeline.py` - Metadata extraction, transcript pipeline, chunking, embeddings, retriever, QA chain
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (local only, never commit)

## How It Works

1. Video metadata is fetched with `yt-dlp`.
2. Transcript is fetched (API first, Whisper fallback if needed).
3. Transcript is split into timestamp-aware chunks with metadata headers.
4. Chunks are embedded using `gemini-embedding-001` and stored in FAISS.
5. Hybrid retrieval (`FAISS + BM25`) selects the best context.
6. Groq model generates grounded answers with timestamp citations.

## Prerequisites

- Python `3.11` recommended
- Git
- Internet access
- API keys for Groq and Google AI Studio

## Local Setup

### 1) Clone repository

```bash
git clone <your-repo-url>
cd youtube
```

### 2) Create and activate virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4) Create `.env`

```env
# Required
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key

# Optional (LangSmith tracing)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=youtube-qa
```

### 5) Run app

```bash
streamlit run app.py --server.fileWatcherType none --server.address 127.0.0.1 --server.port 8501
```

Open: `http://127.0.0.1:8501`

## API Keys

### Groq

- Console: [https://console.groq.com](https://console.groq.com)
- Used for chat generation and cloud Whisper fallback

### Google AI Studio

- Key page: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- Used for embeddings (`models/gemini-embedding-001`)

## Performance Notes

- Short/medium videos with subtitles are usually fast.
- Long videos create more chunks and higher embedding/token usage.
- First query after restart may be slower due to initialization.
- Keeping the process warm improves consistency.

## Quota and Limits (Important)

If you hit this error during embedding:

- `429 RESOURCE_EXHAUSTED`

It means your Gemini quota/rate limit is exhausted for the current window.

Recommended actions:

1. Retry after cooldown
2. Use shorter videos
3. Add paid quota / billing
4. Add batching + retry logic to embedding calls
5. Add fallback embedding provider (optional future enhancement)

## Common Warnings

### yt-dlp JavaScript runtime warning

You may see:

- `No supported JavaScript runtime could be found...`

This is common and non-blocking for many videos. Some formats may still fail.
Reference: [yt-dlp EJS docs](https://github.com/yt-dlp/yt-dlp/wiki/EJS)

## Troubleshooting

### Blank page in browser

```bash
streamlit cache clear
streamlit run app.py --server.fileWatcherType none --server.address 127.0.0.1 --server.port 8501
```

Then hard refresh (`Cmd + Shift + R`) or try Incognito.

### `GOOGLE_API_KEY is missing`

Add it to `.env`, restart Streamlit.

### Slow processing

- Prefer videos with available subtitles
- Avoid very long videos on free-tier keys
- Expect fallback transcription (Whisper) to be slower than transcript API

## Deployment

## Option A: Streamlit Community Cloud (Fastest)

1. Push code to GitHub
2. Create app in Streamlit Cloud
3. Set entrypoint to `app.py`
4. Add secrets:
   - `GROQ_API_KEY`
   - `GOOGLE_API_KEY`
   - Optional LangSmith keys
5. Deploy

## Option B: Render / Railway

- Runtime: Python 3.11
- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

- Add environment variables in dashboard

## Security Checklist

- Never commit `.env`
- Rotate exposed keys immediately
- Monitor usage in Groq and Google dashboards

## Recommended `.gitignore`

```gitignore
.venv/
.env
__pycache__/
*.pyc
_temp_audio/
.streamlit/
```

## Roadmap

- Embedding batching + exponential retry
- Persistent FAISS cache by `video_id`
- Multi-provider embedding fallback
- Async/background processing for long videos

## License

Choose and add your license (for example, MIT) before open-source release.
