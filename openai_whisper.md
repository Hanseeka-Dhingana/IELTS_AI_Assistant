# What is the Whisper?
Whisper is an open-source automatic speech recognition (ASR) model developed by OpenAI. It can transcribe spoken language into text and also perform translation between languages.

## Core Idea
Whisper is a deep learning model trained on 680,000 hours of multilingual and multitask supervised data collected from the web. This massive training dataset enables it to perform robustly on:

- Transcription (speech → text)
- Translation (non-English speech → English text)
- Language Identification
- Multilingual transcription
- Speech detection

## 🏗️ How Whisper Works (Architecture Insight)
Whisper uses a transformer-based encoder-decoder architecture, similar to models like GPT or T5. Here's a simplified breakdown:

**🧱 1. Audio Input**   
- Audio is broken into 30-second chunks
- It's resampled to 16 kHz
- Converted into log-Mel spectrograms (a visual representation of audio)

**🔁 2. Encoder**  
Processes the spectrogram to extract latent audio features
  
**📤 3. Decoder**  
- Predicts the text output token-by-token
- It is trained autoregressively, meaning it uses previous tokens to predict the next

**🔄 Tasks It Can Perform:**
- transcribe: Audio → Text in the same language
- translate: Audio → English text from any language  

<br>  

## 🔎 Key Features
| Feature                     | Description                                                |
| --------------------------- | ---------------------------------------------------------- |
| **Robust to accents**       | Trained on diverse speakers, so works with various accents |
| **Multilingual**            | Handles transcription and translation across 96+ languages |
| **No need for fine-tuning** | General-purpose, works out-of-the-box for many tasks       |
| **Open-source**             | Fully open-source under MIT license                        

<br>  

## 💬 Whisper Model Variants  
| Model Name | Size  | Speed     | Accuracy  |
| ---------- | ----- | --------- | --------- |
| `tiny`     | 39M   | Very fast | Lower     |
| `base`     | 74M   | Fast      | Moderate  |
| `small`    | 244M  | Medium    | Good      |
| `medium`   | 769M  | Slower    | Very good |
| `large`    | 1550M | Slowest   | Best      |
     
<br>  

## ⚙️ Example Code
``` 
   pip install openai-whisper
```
``` python 
   import whisper

   model = whisper.load_model("base")  # You can use 'small', 'medium', etc.
   result = model.transcribe("audio.mp3")
   print(result["text"])
```
**Translation:**  
``` python 
    result = model.transcribe("audio.mp3", task="translate")
    print(result["text"])  # Translates to English
```

<br>  

## 🎯 Use Case of Whisper in IELTS Chatbot
Whisper by OpenAI in your IELTS chatbot adds a powerful speaking test and speech recognition feature — one of the four main components of the IELTS exam.    

**Purpose:**   
To simulate the IELTS Speaking test by allowing users to speak their answers and have them:

1. Transcribed into text
2. Analyzed for grammar, fluency, and coherence
3. Scored and given feedback

# Sources:
**Whisper (OpenAI)**  
https://www.gladia.io/blog/what-is-openai-whisper
https://openai.com/index/whisper/  

https://medium.com/%40okezieowen/whisper-functionality-and-finetuning-ba7f9444f55a

**log-Mel spectrograms**  
https://www.mathworks.com/help/audio/ref/melspectrogram.html

https://huggingface.co/learn/audio-course/en/chapter1/audio_data

**Generating Subtitles with OpenAI Whisper**  
https://medium.com/akvelon/generating-subtitles-for-youtube-videos-with-openai-whisper-72b1f9a594ea



