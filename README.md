# Study App

A local study tool to process materials and generate revision workflows. Built with Python and React.

## Features

- Hybrid RAG Integration: Ingest multiple PDF documents using a dual retrieval strategy.
- Two-Layer Anti-Hallucination: Strict relevance gates ensure the model answers only from your uploaded documents.
- Flashcard Generator: Extract key concepts to create term/definition pairs.
- Revision Planner: Analyze documents and generate study schedules.
- Grounded Quiz: Generate custom multiple-choice quizzes that cite source files.

## Technology Stack

- Frontend: React.js
- Backend API: FastAPI
- Orchestration: LangChain
- Embeddings: Local nomic-embed-text via Ollama
- Vector Database: ChromaDB
- Keyword Retrieval: BM25

## Getting Started

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

## Usage

1. **Upload Study Materials**: Use the upload area on the left to select and submit your PDF files.
2. **Process PDFs**: Click the process button to ingest, chunk, and embed your documents into the local knowledge base.
3. **Choose a Tool**: Select Flashcards, Revision Planner, or Grounded Quiz.
4. **Generate**: Enter a topic and click Generate. The system will first check if the topic exists in your PDFs before answering.


## Caching
This system uses in-memory dictionaries on the backend to optimize performance. Relevance checks are cached so the same topic doesn't need to be re-evaluated, keeping the frontend responsive. Caches reset when the backend is restarted.
