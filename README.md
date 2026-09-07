# IELTS AI Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io/)

An AI-powered IELTS assistant that helps learners **ask questions about IELTS** and **practice IELTS Speaking** through an interactive web interface.

The project combines **Retrieval-Augmented Generation (RAG)** with speech processing to provide relevant IELTS information and personalized speaking feedback.

## 🚀 Live UI

**Try the IELTS AI Assistant:**
[IELTS AI Assistant](https://ieltsaiassistant.streamlit.app/)

---


## 📌 What the Project Does

IELTS AI Assistant provides two main functionalities:

### 1. 💬 IELTS Q&A Chatbot

Users can ask questions related to IELTS, such as:

* IELTS exam format
* Speaking test structure
* Band scores and criteria
* IELTS preparation
* Speaking strategies
* Grammar and vocabulary questions
* General IELTS-related guidance

The chatbot uses **RAG** to retrieve relevant information from the project's IELTS knowledge base before generating an answer.

### 2. 🎤 IELTS Speaking Practice

The speaking interface allows users to practice IELTS Speaking by providing their answer through voice input.

The system:

1. Captures the user's speech
2. Converts speech to text using **Whisper**
3. Processes the response
4. Evaluates speaking-related characteristics
5. Generates feedback using AI

This helps learners understand their performance and identify areas for improvement.

---

## ✨ Key Features

* 💬 **IELTS Q&A Chatbot** for IELTS-related questions
* 📚 **Retrieval-Augmented Generation (RAG)** for knowledge-grounded answers
* 🔎 **Semantic search** using vector embeddings
* 🗄️ **Pinecone vector database** for storing and retrieving IELTS knowledge
* 🧠 **Gemini LLM** for generating responses and feedback
* 🎤 **Voice input** for IELTS Speaking practice
* 🗣️ **Whisper Speech-to-Text** for transcribing spoken answers
* 📊 **BERT-based assessment** for speaking analysis
* 🌐 **Streamlit web interface**
* ⚡ Combines generative and discriminative AI for IELTS assistance

---

## 🏗️ System Architecture

The project contains two main AI workflows.

### IELTS Q&A Chatbot

```text
User Question
      ↓
Query Processing
      ↓
Generate Embedding
      ↓
Pinecone Vector Search
      ↓
Retrieve Relevant IELTS Content
      ↓
Gemini LLM
      ↓
AI-Generated Answer
      ↓
User
```

### IELTS Speaking

```text
User Voice
     ↓
Whisper Speech-to-Text
     ↓
Transcribed Response
     ↓
Speaking Analysis
     ↓
BERT / AI Assessment
     ↓
Feedback
     ↓
User
```

---

## 🧠 Retrieval-Augmented Generation

The chatbot uses **Retrieval-Augmented Generation (RAG)** instead of relying only on the language model's internal knowledge.

The process is:

1. IELTS resources are processed into documents.
2. Documents are divided into smaller chunks.
3. Embeddings are generated for the chunks.
4. Embeddings are stored in Pinecone.
5. When a user asks a question, the question is converted into an embedding.
6. Pinecone retrieves the most relevant IELTS information.
7. The retrieved context is provided to Gemini.
8. Gemini generates the final response.

This helps the chatbot provide answers based on the project's IELTS knowledge base.

---

## 🛠️ Tech Stack

| Technology            | Purpose                               |
| --------------------- | ------------------------------------- |
| **Python**            | Core application development          |
| **Streamlit**         | Web interface                         |
| **LlamaIndex**        | RAG and document indexing             |
| **Pinecone**          | Vector database and similarity search |
| **Google Gemini**     | LLM and AI-generated responses        |
| **Gemini Embeddings** | Text embeddings                       |
| **Whisper**           | Speech-to-text                        |
| **BERT**              | Speaking assessment                   |
---

## 📂 Project Structure

A typical project structure is:

```text
IELTS_AI_Assistant/
│
├── app.py
├── chatbot_logic.py
├── requirements.txt
├── .env
│
├── data/
│   └── IELTS resources
│
├── ...
│
└── README.md
```

> The exact file structure may vary depending on the current version of the project.

---

## ⚙️ Getting Started

### Prerequisites

Make sure you have:

* Python 3.10 or newer
* A Google Gemini API key
* A Pinecone API key
* A Pinecone index
* Internet connection

### 1. Clone the Repository

```bash
git clone https://github.com/Hanseeka-Dhingana/IELTS_AI_Assistant.git
cd IELTS_AI_Assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Do **not** commit your `.env` file or API keys to GitHub.

### 5. Configure Pinecone

The RAG system uses a Pinecone vector index for IELTS documents.

The embedding configuration used by the project should match the vector index configuration, including its embedding dimension and similarity metric.

For example:

```text
Dimension: 768
Metric: cosine
```

### 6. Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

---

## 💡 Usage

### Ask an IELTS Question

Enter a question into the chatbot, for example:

```text
What are the IELTS Speaking band score criteria?
```

The system retrieves relevant information from the IELTS knowledge base and generates an answer.

### Practice Speaking

Use the speaking interface to provide an IELTS Speaking response through voice input.

The system converts the response into text and uses the AI assessment pipeline to provide feedback.

---

## 🔐 Environment Variables

The application requires API credentials for external AI/vector services.

| Variable           | Description           |
| ------------------ | --------------------- |
| `GEMINI_API_KEY`   | Google Gemini API key |
| `PINECONE_API_KEY` | Pinecone API key      |

Keep all credentials private.

---

## 📖 Why RAG?

RAG is useful for this project because IELTS information can be stored in a dedicated knowledge base and retrieved when needed.

Instead of asking the LLM to answer entirely from its pretrained knowledge:

```text
User Question
      ↓
Retrieve IELTS Information
      ↓
LLM + Retrieved Context
      ↓
Answer
```

This makes the chatbot more suitable for answering questions based on the project's IELTS resources.

---

## 🎯 Project Goals

The main goal of IELTS AI Assistant is to provide an accessible AI-based tool that helps IELTS learners:

* Get quick answers to IELTS-related questions
* Practice IELTS Speaking
* Receive automated feedback
* Use voice-based interaction
* Learn using an IELTS-focused knowledge base

---

## 🔮 Future Improvements

Potential improvements include:

* More detailed IELTS Speaking scoring
* Improved pronunciation analysis
* Better fluency and coherence evaluation
* Additional IELTS resources
* Conversation history
* Improved RAG retrieval and ranking
* More personalized speaking feedback
* Better voice interaction
* Deployment optimization

---

## 🤝 Contributing

Contributions are welcome.

If contribution guidelines are available in the repository, please refer to:

[`CONTRIBUTING.md`](CONTRIBUTING.md)

A typical contribution workflow is:

```bash
git checkout -b feature/your-feature
```

Make your changes, test them locally, and submit a pull request.

---

## 🆘 Getting Help

For project-related questions or issues:

* Open an issue in the GitHub repository.
* Check the project documentation if available.
* Review the setup instructions above before running the application.

---

## 👤 Maintainer

**Hanseeka Dhingana**

GitHub: [@Hanseeka-Dhingana](https://github.com/Hanseeka-Dhingana)

Project: [IELTS AI Assistant](https://github.com/Hanseeka-Dhingana/IELTS_AI_Assistant)
