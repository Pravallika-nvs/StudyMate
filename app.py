import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import random, json, tempfile, os
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
import google.generativeai as genai
from deep_translator import GoogleTranslator, single_detection
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av, speech_recognition as sr

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="StudyMate", page_icon="📘", layout="wide")
st.title("📘 StudyMate - AI Study Assistant")

# Hugging Face & Gemini setup
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
client = InferenceClient(model=HF_MODEL)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ------------------------------
# SESSION STATE
# ------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "chunk_metadata" not in st.session_state:
    st.session_state.chunk_metadata = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

# ------------------------------
# FUNCTIONS
# ------------------------------
def extract_chunks_with_metadata(pdf_file, chunk_size=1000, overlap=200):
    chunks, metadata = [], []
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        for start in range(0, len(text), chunk_size - overlap):
            chunk = text[start:start+chunk_size].strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({"pdf": uploaded_file.name, "page": page_num})
    return chunks, metadata

def generate_answer(query, chunks, chunk_metadata, top_k=3):
    """Smart answer generation with PDF citation & Gemini fallback."""
    if not chunks:
        gemini_response = gemini_model.generate_content(query)
        return gemini_response.text + "\n\n💡 No PDF uploaded.", False, None

    # Embed and search top PDF chunks
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    relevant_chunks, context = [], ""
    for rank, idx in enumerate(indices[0]):
        relevant_chunks.append({
            "text": chunks[idx],
            "pdf": chunk_metadata[idx]["pdf"],
            "page": chunk_metadata[idx]["page"],
            "distance": distances[0][rank]
        })
        context += f"[{chunk_metadata[idx]['pdf']} - Page {chunk_metadata[idx]['page']}]\n{chunks[idx]}\n\n"

    # Decide if PDF is relevant
    best_distance = distances[0][0]

    # Case 1: Relevant PDF Context
    if best_distance < 1.0:
        try:
            messages = [
                {"role": "system", "content": "You are an academic assistant. Use the PDF context and cite the source."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
            ]
            response = client.chat_completion(messages=messages, max_tokens=300)
            answer = response.choices[0].message["content"]
            top_source = relevant_chunks[0]
            return f"{answer}\n\n📄 Answered using **{top_source['pdf']}**, page {top_source['page']}.", True, relevant_chunks
        except HfHubHTTPError:
            st.warning("⚠️ Hugging Face failed, using Gemini fallback...")
            prompt = f"Context:\n{context}\n\nQuestion:\n{query}\nAnswer and cite the PDF pages."
            gemini_response = gemini_model.generate_content(prompt)
            top_source = relevant_chunks[0]
            return f"{gemini_response.text}\n\n📄 (Gemini) Used **{top_source['pdf']}**, page {top_source['page']}.", True, relevant_chunks

    # Case 2: No Relevant PDF Context
    gemini_prompt = f"""
    You are a helpful academic AI tutor. 
    Question: {query}

    - No relevant PDF content found.
    - Answer normally and clearly.
    """
    gemini_response = gemini_model.generate_content(gemini_prompt)
    return gemini_response.text + "\n\n💡 This question was answered outside the uploaded PDFs.", False, None

def generate_mcqs_from_chunks(chunks, num_mcqs=5):
    all_sentences = [line.strip() for chunk in chunks for line in chunk.split(". ") if len(line.strip()) > 20]
    selected = random.sample(all_sentences, min(num_mcqs, len(all_sentences)))
    mcqs = []
    for q in selected:
        mcqs.append({
            "question": q,
            "options": random.sample(all_sentences, 4) if len(all_sentences) >= 4 else [q]*4,
            "answer": q,
            "concept": "General",
            "explanation": q
        })
    return mcqs

def speak_text(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("📄 Upload a PDF", type="pdf")
if uploaded_file:
    st.session_state.chunks, st.session_state.chunk_metadata = extract_chunks_with_metadata(uploaded_file)
    st.success(f"✅ Extracted {len(st.session_state.chunks)} chunks from {uploaded_file.name}")

# ------------------------------
# CHAT INTERFACE
# ------------------------------
prompt_text = st.chat_input("Ask a question (PDF or general)...")

# Voice input (WebRTC)
st.markdown("### 🎤 Voice Input")
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.audio_data = bytearray()
        self.recognized_text = None
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_data.extend(frame.to_ndarray().tobytes())
        return frame
    def get_text(self):
        if len(self.audio_data) > 0:
            with sr.AudioFile(self._to_wav()) as source:
                audio = self.recognizer.record(source)
            try:
                self.recognized_text = self.recognizer.recognize_google(audio)
            except:
                self.recognized_text = None
            self.audio_data = bytearray()
        return self.recognized_text
    def _to_wav(self):
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_wav.name, "wb") as f:
            f.write(self.audio_data)
        return temp_wav.name

webrtc_ctx = webrtc_streamer(
    key="speech", mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)
if webrtc_ctx.audio_receiver:
    audio_processor = AudioProcessor()
    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
    for frame in audio_frames:
        audio_processor.recv_audio(frame)
    spoken_text = audio_processor.get_text()
    if spoken_text:
        prompt_text = spoken_text
        st.success(f"🎤 Recognized: {spoken_text}")

# Handle chat
if prompt_text:
    input_lang = single_detection(prompt_text, api_key=None)
    translated_q = GoogleTranslator(source='auto', target='en').translate(prompt_text)
    answer, from_pdf, sources = generate_answer(translated_q, st.session_state.chunks, st.session_state.chunk_metadata)
    answer_final = GoogleTranslator(source='en', target=input_lang).translate(answer)

    st.chat_message("user").markdown(prompt_text)
    st.chat_message("assistant").markdown(answer_final)
    st.session_state.chat_history.append({"role": "user", "content": prompt_text})
    st.session_state.chat_history.append({"role": "assistant", "content": answer_final})

    if st.checkbox("🔊 Speak Answer"):
        st.audio(speak_text(answer_final, lang=input_lang), format="audio/mp3")

# ------------------------------
# QUIZ SECTION
# ------------------------------
if st.session_state.chunks and not st.session_state.quiz_questions:
    st.session_state.quiz_questions = generate_mcqs_from_chunks(st.session_state.chunks, num_mcqs=5)

if st.session_state.quiz_questions:
    with st.expander("🧠 Test Your Knowledge - Quiz", expanded=False):
        for i, item in enumerate(st.session_state.quiz_questions, start=1):
            qkey = f"q{i}"
            st.markdown(f"**Q{i}:** {item['question']}?")
            st.session_state.quiz_answers[qkey] = st.radio(
                label=f"Select answer for Q{i}",
                options=item['options'],
                index=0,
                key=qkey,
                label_visibility="collapsed"
            )
        if st.button("✅ Submit Quiz"):
            score = sum(1 for i, item in enumerate(st.session_state.quiz_questions, start=1)
                        if st.session_state.quiz_answers[f"q{i}"] == item['answer'])
            st.success(f"Your Score: {score}/{len(st.session_state.quiz_questions)}")
            st.session_state.quiz_submitted = True
        if st.button("🔁 Retake Quiz"):
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False

# ------------------------------
# DOWNLOAD CHAT
# ------------------------------
if st.session_state.chat_history:
    st.download_button(
        label="⬇️", 
        data=json.dumps(st.session_state.chat_history, indent=2),
        file_name="chat_history.json",
        mime="application/json",
        help="Download Chat"
    )
