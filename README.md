# 📚 Exam Revision Assistant

A fully local AI-powered study companion designed to process your study materials and generate interactive revision workflows without compromising your data privacy. Built with **Ollama**, **LangChain**, and **Streamlit**.

## ✨ Features

- **Local RAG Integration**: Ingest multiple PDF documents (syllabuses, textbooks, lecture slides) securely on your own hardware using ChromaDB vector persistence.
- **🗂️ Flashcard Generator**: Automatically extract key concepts from your materials to create high-yield JSON term/definition pairs that can be downloaded straight to your machine.
- **🗓️ Revision Planner**: Analyze your syllabus documents and generate structurally prioritized, markdown-formatted study schedules.
- **🎯 Grounded Quiz**: Generate custom multiple-choice quizzes that cite the exact source file and page numbers for the correct answers.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://python.langchain.com/) for chunking, routing, and creating standard RAG chains.
- **LLM Engine**: [Ollama](https://ollama.com/) (using `llama3.1:8b` for high-quality generation and `nomic-embed-text` for vector embeddings)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (stored locally in `./exam_knowledge_db`)

## 🚀 Getting Started

### Prerequisites

1. Download and install **[Ollama](https://ollama.com/)** on your operating system.
2. An active installation of Python 3.9+.

### Installation

1. **Clone or navigate** to the project directory.
2. **Execute the Environment Setup Script:** 
   Open PowerShell and run the integrated setup script. This script will upgrade `pip`, install all necessary python dependencies from `requirements.txt`, and automatically pull your required machine learning models down to your hardware.
   
   ```powershell
   ./environment_setup.ps1
   ```

### Running the Application

Because of the way Windows registers Python packages on PATH, the recommended way to start your dashboard is via the Python module:

```bash
python -m streamlit run app.py
```

This will launch the dashboard locally at `http://localhost:8501`. 

## 📖 Usage

1. Open the left sidebar and upload multiple PDF files relevant to your upcoming exams.
2. Click **Process PDFs**. You will see the application ingest, chunk, and embed your vectors into ChromeDB.
3. In the main interface, select whichever tool you want to use (Flashcards, Planner, or Quiz).
4. Enter the overarching topic you are focusing on and press **Generate**!

## 💾 Caching
This system features Streamlit `@st.cache_data`. When you ask for the same topic's generated output twice, it will load instantly to save GPU processing power. This cache automatically resets when new PDFs are ingested.
