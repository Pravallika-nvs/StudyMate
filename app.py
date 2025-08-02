import streamlit as st
from deep_translator import GoogleTranslator
from huggingface_hub import InferenceClient
from transformers import pipeline
import torch
from gtts import gTTS
import tempfile
import av
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="StudyMate", page_icon="📘", layout="wide")
st.title("📘 StudyMate - Multilingual PDF Q&A")

# Hugging Face API Client
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your model
client = InferenceClient(model=HF_MODEL)

# Local Fallback (tiny model for offline/failover)
fallback_model = pipeline("text-generation", model="distilgpt2", device=0 if torch.cuda.is_available() else -1)

# ------------------------------
# SESSION STATE
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "voice_text" not in st.session_state:
    st.session_state.voice_text = None

# ------------------------------
# FUNCTIONS
# ------------------------------

def generate_grounded_answer(question):
    """Try Hugging Face API first, fallback to local model if 503 occurs."""
    messages = [
        {"role": "system", "content": "You are an academic assistant. Answer clearly and concisely."},
        {"role": "user", "content": question}
    ]
    try:
        response = client.chat_completion(messages=messages, max_tokens=300)
        return response.choices[0].message["content"]
    except Exception:
        st.warning("⚠️ Hugging Face API unavailable, switching to fallback model.")
        local_resp = fallback_model(question, max_length=200, do_sample=True)
        return local_resp[0]["generated_text"]

def speak_text(text):
    tts = gTTS(text=text, lang="en")
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name

def download_chat():
    chat_text = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history])
    st.download_button(
        label="⬇️", 
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain",
        help="Download Chat"
    )

# ------------------------------
# WebRTC Audio Processor for Voice Input
# ------------------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.audio_data = bytearray()
        self.recognized_text = None

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_data.extend(audio.tobytes())
        return frame

    def get_text(self):
        if len(self.audio_data) > 0:
            with sr.AudioFile(self._to_wav()) as source:
                audio = self.recognizer.record(source)
            try:
                self.recognized_text = self.recognizer.recognize_google(audio)
            except:
                self.recognized_text = None
            self.audio_data = bytearray()  # reset
        return self.recognized_text

    def _to_wav(self):
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with open(temp_wav.name, "wb") as f:
            f.write(self.audio_data)
        return temp_wav.name

# ------------------------------
# UI
# ------------------------------

col1, col2 = st.columns([8,1])
with col1:
    prompt = st.chat_input("Ask a question...")
with col2:
    download_chat()

st.markdown("### 🎤 Voice Input")
webrtc_ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDONLY,
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
        st.session_state.voice_text = spoken_text
        st.success(f"Voice recognized: **{spoken_text}**")
        prompt = spoken_text

# ------------------------------
# Handle Q&A
# ------------------------------
if prompt:
    translated_q = GoogleTranslator(source='auto', target='en').translate(prompt)
    answer = generate_grounded_answer(translated_q)
    answer_final = GoogleTranslator(source='en', target='auto').translate(answer)

    # Save chat history
    st.session_state.chat_history.append(("user", prompt))
    st.session_state.chat_history.append(("assistant", answer_final))

    # Display chat
    st.chat_message("user").markdown(prompt)
    st.chat_message("assistant").markdown(answer_final)

    # Voice Output
    if st.checkbox("🔊 Speak Answer"):
        audio_file = speak_text(answer_final)
        st.audio(audio_file, format="audio/mp3")

# Display previous chat
for role, msg in st.session_state.chat_history:
    st.chat_message(role).markdown(msg)
