import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import streamlit as st
from rag_pipeline import process_youtube_url, generate_chapters

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Q&A",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 YouTube Video Q&A")
st.markdown("Paste a YouTube URL and ask any question about the video!")

# ── Clickable Timestamp Links (Feature #4) ────────────────────────
def make_timestamps_clickable(text, video_id):
    """
    Convert [MM:SS] timestamps in text to clickable YouTube links.
    Example: [12:35] → [12:35](https://youtube.com/watch?v=VIDEO_ID&t=755)
    """
    def replace_timestamp(match):
        time_str = match.group(1)  # "12:35"
        parts = time_str.split(":")
        total_seconds = int(parts[0]) * 60 + int(parts[1])
        url = f"https://youtube.com/watch?v={video_id}&t={total_seconds}"
        return f"[▶ {time_str}]({url})"

    return re.sub(r'\[(\d{2}:\d{2})\]', replace_timestamp, text)

# ── Local Pre-Filter (saves API tokens) ──────────────────────────
SELF_PATTERNS = [
    r'\bwho\s+are\s+you\b', r'\bwhat\s+are\s+you\b', r'\byour\s+name\b',
    r'\bwhat\s+model\b', r'\bwhich\s+model\b', r'\btrained\s+on\b',
    r'\bwhat\s+can\s+you\s+do\b', r'\btell\s+me\s+about\s+yourself\b',
    r'\bwho\s+made\s+you\b', r'\bwho\s+created\s+you\b', r'\bwho\s+built\s+you\b',
    r'\bare\s+you\s+ai\b', r'\bare\s+you\s+a\s+bot\b', r'\bare\s+you\s+human\b',
    r'\byour\s+training\b', r'\byour\s+data\b', r'\byour\s+purpose\b',
    r'\bwhat\s+language\s+model\b', r'\bwhat\s+llm\b', r'\bgpt\b', r'\bchatgpt\b',
    r'\bgemini\b', r'\bclaude\b', r'\byou\s+trained\b',
]

OFFTOPIC_PATTERNS = [
    r'^\s*\d+\s*[\+\-\*\/\%\^]\s*\d+',       # math: "1+1", "5*3"
    r'\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+',  # "what is 1+20"
    r'\bcalculate\b', r'\bsolve\b',
    r'\bwrite\s+(a\s+)?code\b', r'\bwrite\s+(a\s+)?program\b',
    r'\bpython\b', r'\bjava\b', r'\bjavascript\b', r'\bc\+\+\b',
    r'\bhtml\b', r'\bcss\b', r'\bsql\b',
    r'\brecipe\b', r'\bweather\b', r'\bstock\s+price\b',
    r'\btell\s+me\s+a\s+joke\b', r'\bsing\b', r'\bwrite\s+a\s+poem\b',
]

def check_off_topic(question):
    """Check if a question is off-topic. Returns rejection message or None."""
    q = question.lower().strip()
    for pattern in SELF_PATTERNS:
        if re.search(pattern, q):
            return "I can only answer questions about this video. Please ask something related to the video content."
    for pattern in OFFTOPIC_PATTERNS:
        if re.search(pattern, q):
            return "I can only answer questions about this video. That topic is not discussed here."
    return None

# ── Session State ─────────────────────────────────────────────────
for key, default in {
    "qa_chain": None, "chat_history": [], "chat_history_for_langchain": [], "video_metadata": None,
    "transcript": None, "video_id": None, "chapters": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Section 1: Load Video ─────────────────────────────────────────
st.subheader("📎 Step 1 — Paste a YouTube URL")
url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("🚀 Process Video", type="primary"):
    if url:
        with st.status("Processing video...", expanded=True) as status:
            try:
                st.write("📡 Fetching video metadata...")
                import time
                t_start = time.time()

                qa_chain, num_chunks, metadata, transcript, video_id = process_youtube_url(url)

                elapsed = time.time() - t_start
                status.update(label=f"✅ Done in {elapsed:.0f}s!", state="complete", expanded=False)

                st.session_state.qa_chain = qa_chain
                st.session_state.video_metadata = metadata
                st.session_state.transcript = transcript
                st.session_state.video_id = video_id
                st.session_state.chat_history = []
                st.session_state.chat_history_for_langchain = []
                st.session_state.chapters = None
                st.success(f"✅ Created {num_chunks} knowledge chunks in {elapsed:.0f}s. Ask your questions below!")
            except Exception as e:
                status.update(label="❌ Failed", state="error")
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Make sure the video is publicly accessible and has captions/subtitles.")
    else:
        st.warning("⚠️ Please enter a YouTube URL first.")

# ── Section 1b: Video Info Card ───────────────────────────────────
if st.session_state.video_metadata:
    meta = st.session_state.video_metadata
    st.divider()
    st.subheader("📺 Video Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**🎬 Title:** {meta.get('title', 'Unknown')}")
        st.markdown(f"**📺 Channel:** {meta.get('channel', 'Unknown')}")
        st.markdown(f"**🏷️ Category:** {meta.get('category', 'Unknown')}")
    with col2:
        st.markdown(f"**🌐 Language:** {meta.get('language', 'Unknown')}")
        st.markdown(f"**⏱️ Duration:** {meta.get('duration', 'Unknown')}")

# ── Section 1c: Auto Chapter Generator (Feature #1) ──────────────
if st.session_state.transcript and st.session_state.video_metadata:
    st.divider()
    st.subheader("📋 Auto Chapters")

    if st.session_state.chapters:
        # Display cached chapters with clickable timestamps
        chapter_text = st.session_state.chapters
        if st.session_state.video_id:
            chapter_text = make_timestamps_clickable(chapter_text, st.session_state.video_id)
        st.markdown(chapter_text)
    else:
        st.markdown("*Generate an automatic table of contents with timestamps — even for videos without chapters!*")
        if st.button("📋 Generate Chapters", type="secondary"):
            with st.spinner("Analyzing transcript and generating chapters..."):
                try:
                    chapters = generate_chapters(
                        st.session_state.transcript,
                        st.session_state.video_metadata
                    )
                    st.session_state.chapters = chapters
                    # Display with clickable timestamps
                    if st.session_state.video_id:
                        chapters = make_timestamps_clickable(chapters, st.session_state.video_id)
                    st.markdown(chapters)
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "rate_limit" in error_msg.lower():
                        st.warning("⚠️ API rate limit reached. Please wait a few minutes and try again.")
                    else:
                        st.error(f"❌ Could not generate chapters: {error_msg}")

# ── Section 2: Chat ───────────────────────────────────────────────
if st.session_state.qa_chain:
    st.divider()
    st.subheader("💬 Step 2 — Ask Questions About the Video")

    # Display chat history (with clickable timestamps)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant" and st.session_state.video_id:
                content = make_timestamps_clickable(content, st.session_state.video_id)
            st.markdown(content)

    # Chat input box
    question = st.chat_input("Ask anything about the video...")

    if question:
        # Show user question
        with st.chat_message("user"):
            st.write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        # Get and show answer
        with st.chat_message("assistant"):
            rejection = check_off_topic(question)

            if rejection:
                answer = rejection
                st.write(answer)
            else:
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.qa_chain.invoke({
                            "question": question,
                            "chat_history": st.session_state.chat_history_for_langchain
                        })
                        answer = result["answer"]
                        
                        # Add to Langchain's tuple-based memory
                        st.session_state.chat_history_for_langchain.append((question, answer))
                        
                        # Make timestamps clickable in the answer
                        display_answer = answer
                        if st.session_state.video_id:
                            display_answer = make_timestamps_clickable(answer, st.session_state.video_id)
                        st.markdown(display_answer)
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "rate_limit" in error_msg.lower():
                            answer = "⚠️ API rate limit reached. Please wait a few minutes and try again."
                        else:
                            answer = f"⚠️ Error getting answer: {error_msg}"
                        st.warning(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})