import streamlit as st
import google.generativeai as genai
from faster_whisper import WhisperModel
import tempfile
import os
from pydub import AudioSegment
from pydub.effects import normalize


# pytorch is used indirectly by whsiper this line avoid the "RuntimeError: Tried to instantiate class '__path__._path'."
import torch
torch.classes.__path__ = []  # Neutralizes the path inspection  

# Load Whisper
faster_whisper_model = WhisperModel("small", compute_type = "int8")

# Set Gemini API Key
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")


class IELTSSpeakingTest:
    def __init__(self):
        self.transcript = ""
        self.question = self.generate_question()
        
        
    # Generate the question from gemini 
    def generate_question(self):
        prompt = "Generate a random IELTS Speaking part 1 question."
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

     
    # Start function
    def start_test(self):
        st.write("### Speaking Test Started!")
        st.write(f"**Question:** {self.question}")

        audio_data = st.audio_input("Record your answer:")

        if audio_data is not None:
            st.audio(audio_data)

            # Save uploaded audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(audio_data.read())
                tmp_path = tmpfile.name
                
            
            # Preprocess audio: Convert to mono, 16KHz, normalize
            audio = AudioSegment.from_file(tmp_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio = normalize(audio)
            
            Cleaned_path = tmp_path.replace(".wav", "_cleaned.wav")
            audio.export(Cleaned_path, format = "wav") 

            # Transcribe with Faster-Whisper
            segments, _ = faster_whisper_model.transcribe(Cleaned_path)
            
            # Combine all segments into full transcript
            self.transcript = " ".join([seg.text for seg in segments])
            
            # Cleanup temp files
            os.remove(tmp_path)
            os.remove(Cleaned_path)

            st.success("Transcription complete!")
            st.write(f"**Your Answer:**  {self.transcript}")
            self.evaluate_answer()

    def evaluate_answer(self):
        feedback = f"Feedback:"
        st.info("### Feedback")
        st.write(feedback)
        st.write("band score:")
