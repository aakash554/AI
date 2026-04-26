# 📚 Exam Revision Assistant

A local AI-powered study companion designed to process your study materials and generate interactive revision workflows without compromising accuracy. Built with **Gemini 2.5 Flash**, **Ollama**, **LangChain**, **FastAPI**, and **React**.

## ✨ Features

- **Hybrid RAG Integration**: Ingest multiple PDF documents using a dual retrieval strategy — **Semantic Search** (ChromaDB vector embeddings) + **Keyword Search** (BM25) — merged via Reciprocal Rank Fusion (RRF) for highly accurate results.
- **🛡️ Two-Layer Anti-Hallucination**: Strict relevance gates (embedding similarity threshold + LLM verification) ensure the model answers **only** from your uploaded documents. If the information isn't in your PDFs, it explicitly states it is not available.
- **🗂️ Flashcard Generator**: Automatically extract key concepts from your materials to create high-yield term/definition pairs that can be downloaded as JSON.
- **🗓️ Revision Planner**: Analyze your syllabus documents and generate structurally prioritized, markdown-formatted study schedules.
- **🎯 Interactive Grounded Quiz**: Generate custom multiple-choice quizzes with an interactive UI (radio buttons, live scoring, explanations) that cite the exact source file and page numbers for the correct answers.

## 🛠️ Technology Stack

- **Frontend**: [React.js](https://react.dev/) via Vite.
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/)
- **Orchestration**: [LangChain](https://python.langchain.com/) for chunking, routing, and RAG pipelines.
- **LLM Engine**: [Google Gemini 2.5 Flash](https://ai.google.dev/) via `langchain-google-genai` for fast, accurate text generation.
- **Embeddings**: Local `nomic-embed-text` via [Ollama](https://ollama.com/) (Free, completely local embeddings).
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (stored locally in `./exam_knowledge_db`).
- **Keyword Retrieval**: [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) via `rank_bm25` for hybrid search.

## 🚀 Getting Started

### Prerequisites

1. Python 3.9+ installed.
2. [Node.js](https://nodejs.org/) installed (for the React frontend).
3. [Ollama](https://ollama.com/) installed and running.
3. A **Google API Key** for Gemini — get one free at [Google AI Studio](https://aistudio.google.com/apikey).

### Installation

1. **Clone or navigate** to the project directory.
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Pull the Ollama embedding model:**
   ```bash
   ollama pull nomic-embed-text
   ```

### Configuration

Create a `.env` file in the root of the project directory and add your Google API key:

```env
GOOGLE_API_KEY=your-actual-api-key-here
```

*Note: The application will automatically load this key in the background via `python-dotenv`. You do not need to enter it into the frontend UI, keeping it secure.*

### Running the Application

You need to run both the backend API and the frontend UI concurrently.

**1. Start the FastAPI Backend:**
Open a terminal in the root directory:
```bash
uvicorn api:app --reload
```
The API will be available at `http://localhost:8000`.

**2. Start the React Frontend:**
Open a *second* terminal and navigate to the frontend folder:
```bash
cd frontend
npm install
npm run dev
```
The React dashboard will launch at `http://localhost:5173` (or the port specified by Vite).

## 📖 Usage

1. **Upload Study Materials**: Use the upload area on the left to select and submit your PDF files.
2. **Process PDFs**: Click the process button to ingest, chunk, and embed your documents into the local knowledge base.
3. **Choose a Tool**: Select Flashcards, Revision Planner, or Grounded Quiz.
4. **Generate**: Enter a topic and click Generate. The system will first check if the topic exists in your PDFs before answering.
5. **Clear Knowledge Base**: Use the "Clear KB" button to wipe the database when switching to new subjects.

## 💾 Caching
This system uses in-memory dictionaries on the backend to optimize performance. Relevance checks are cached so the same topic doesn't need to be re-evaluated, keeping the frontend responsive. Caches reset when the backend is restarted or the knowledge base is cleared.
