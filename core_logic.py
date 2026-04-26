import os
import json
import pickle
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()  
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.retrievers import BM25Retriever

DB_DIR = "./exam_knowledge_db"
TEMP_DIR = "./temp_uploads"
BM25_DOCS_PATH = os.path.join(DB_DIR, "bm25_docs.pkl")

LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "nomic-embed-text"



def _get_api_key():
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError("GOOGLE_API_KEY is not set.")
    return key

def init_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
        google_api_key=_get_api_key(),
    )

def init_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)


# File & ingestion helpers


def save_uploaded_file(uploaded_file):
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def _save_bm25_docs(splits):
    """Append new document splits to the BM25 doc store."""
    os.makedirs(DB_DIR, exist_ok=True)
    existing = []
    if os.path.exists(BM25_DOCS_PATH):
        with open(BM25_DOCS_PATH, "rb") as f:
            existing = pickle.load(f)
    existing.extend(splits)
    with open(BM25_DOCS_PATH, "wb") as f:
        pickle.dump(existing, f)

def _load_bm25_docs():
    if os.path.exists(BM25_DOCS_PATH):
        with open(BM25_DOCS_PATH, "rb") as f:
            return pickle.load(f)
    return []

def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Metadata normalization
    for doc in docs:
        doc.metadata['source_file'] = os.path.basename(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = init_embeddings()
    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)

    # Persist splits for BM25 keyword retrieval
    _save_bm25_docs(splits)

    return len(splits)

def clear_knowledge_base():
    """Wipe the vector store and BM25 cache so users can start fresh."""
    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)


# Retrieval


def get_vectorstore():
    if not os.path.exists(DB_DIR):
        return None
    try:
        embeddings = init_embeddings()
        vs = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        if vs._collection.count() == 0:
            return None
        return vs
    except Exception:
        return None

def _reciprocal_rank_fusion(results_lists, k=60):
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF)."""
    scores = defaultdict(float)
    doc_map = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc.page_content
            scores[key] += 1.0 / (k + rank + 1)
            doc_map[key] = doc
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[k] for k in sorted_keys]


RELEVANCE_THRESHOLD = 0.5

# Cache relevance verdicts so the same topic returns consistent results
# across flashcards, planner, and quiz within the same session.
_relevance_cache = {}

def _llm_relevance_check(topic: str, context: str) -> bool:
    """Ask the LLM whether the context contains ANY useful info about the topic."""
    cache_key = f"{topic}::{context[:200]}"
    if cache_key in _relevance_cache:
        return _relevance_cache[cache_key]

    llm = init_llm()
    check_prompt = PromptTemplate.from_template(
        """You are a relevance judge. Given a TOPIC and a CONTEXT, decide
whether the CONTEXT contains ANY useful information related to the TOPIC.

RULES:
- Answer ONLY with the single word YES or NO.
- Answer YES if the context discusses the topic, mentions related concepts,
  or contains information that could be used to answer questions about the topic.
- Answer NO only if the context is completely unrelated to the topic —
  for example, asking about "cooking recipes" when the context is about "databases".

TOPIC: {topic}

CONTEXT:
{context}

ANSWER (YES or NO):"""
    )
    chain = check_prompt | llm | StrOutputParser()
    answer = chain.invoke({"topic": topic, "context": context}).strip().upper()
    result = answer.startswith("YES")
    _relevance_cache[cache_key] = result
    return result

def retrieve_with_relevance_check(topic: str):
    """Retrieve docs via hybrid search, then gate on relevance.

    Two-layer check:
      1. Embedding similarity score must exceed RELEVANCE_THRESHOLD.
      2. LLM must confirm the context is genuinely about the topic.

    Returns (docs, is_relevant).
    """
    vs = get_vectorstore()
    if not vs:
        return [], False

    #Layer 1: Embedding similarity score
    scored_results = vs.similarity_search_with_relevance_scores(topic, k=5)
    if not scored_results:
        return [], False

    best_score = max(score for _, score in scored_results)
    if best_score < RELEVANCE_THRESHOLD:
        return [], False          # topic is NOT in the PDFs

    semantic_docs = [doc for doc, _ in scored_results]

    #Hybrid: add BM25 keyword results and fuse 
    bm25_docs_store = _load_bm25_docs()
    if bm25_docs_store:
        bm25_retriever = BM25Retriever.from_documents(bm25_docs_store, k=5)
        bm25_results = bm25_retriever.invoke(topic)
        fused = _reciprocal_rank_fusion([semantic_docs, bm25_results])
        final_docs = fused[:5]
    else:
        final_docs = semantic_docs[:5]

    #  Layer 2: LLM relevance verification
    context_text = "\n".join(d.page_content[:300] for d in final_docs)
    if not _llm_relevance_check(topic, context_text):
        return [], False          # LLM says context is not about the topic

    return final_docs, True

def format_docs(docs):
    if not docs:
        return "NO CONTEXT AVAILABLE — no relevant information found in uploaded documents."
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file','Unknown')} | "
        f"Page {d.metadata.get('page','N/A')}]\n{d.page_content}"
        for d in docs
    )


# Strict grounding instruction injected into every prompt


GROUNDING_RULES = """CRITICAL INSTRUCTIONS — FOLLOW WITHOUT EXCEPTION:
1. Answer ONLY using the CONTEXT provided below. Do NOT use prior knowledge.
2. If the CONTEXT does not contain the answer, respond EXACTLY with:
   " The requested information is not available in the uploaded documents."
3. Do NOT assume, guess, infer, or add anything beyond what is explicitly in the CONTEXT.
4. Always cite the source file name and page number for every piece of information.
5. If the context is only partially relevant, answer only the supported parts and
   clearly state what is missing.

"""

NOT_AVAILABLE = " The requested information is not available in the uploaded documents."

def generate_flashcards(topic: str):
    docs, relevant = retrieve_with_relevance_check(topic)
    if not relevant:
        return [{"term": "Not Available", "definition": NOT_AVAILABLE}]

    llm = init_llm()
    context = format_docs(docs)

    prompt = PromptTemplate.from_template(
        GROUNDING_RULES +
        """You are an exam revision assistant.
Based ENTIRELY on the following CONTEXT, extract key concepts and generate
flashcards for the topic: "{topic}".

RULES:
- ONLY create flashcards from information explicitly present in the CONTEXT.
- Do NOT generate flashcards about topics not covered in the CONTEXT.
- If the CONTEXT has no relevant information, return:
  [{{"term":"Not Available","definition":"The uploaded documents do not contain information about this topic."}}]

Output STRICTLY as a JSON array of objects with keys "term" and "definition".
Include the source citation inside each definition.

Example:
[
  {{"term":"Database","definition":"An organized collection of data. (Source: textbook.pdf, Page 5)"}},
  {{"term":"SQL","definition":"Structured Query Language. (Source: textbook.pdf, Page 12)"}}
]

CONTEXT:
{context}
"""
    )

    chain = prompt | llm | JsonOutputParser()

    try:
        return chain.invoke({"context": context, "topic": topic})
    except Exception as e:
        return {"error": f"Failed to generate flashcards: {str(e)}"}


def generate_revision_planner(topic: str):
    docs, relevant = retrieve_with_relevance_check(topic)
    if not relevant:
        return NOT_AVAILABLE

    llm = init_llm()
    context = format_docs(docs)

    prompt = PromptTemplate.from_template(
        GROUNDING_RULES +
        """You are a revision planner assistant.
Based EXCLUSIVELY on the following CONTEXT, create a structured Markdown study
schedule for the topic: "{topic}".

RULES:
- ONLY include sub-topics explicitly mentioned in the CONTEXT.
- Use weightage/importance hints from the CONTEXT to prioritize if available.
- Do NOT add topics not covered in the CONTEXT.
- Cite source documents for each section.
- If the CONTEXT has no relevant information, respond with:
  " The uploaded documents do not contain sufficient information about '{topic}' to create a study plan."

CONTEXT:
{context}

Markdown Schedule:"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "topic": topic})


def generate_grounded_quiz(topic: str):
    docs, relevant = retrieve_with_relevance_check(topic)
    if not relevant:
        return {"error": NOT_AVAILABLE}

    llm = init_llm()
    context = format_docs(docs)

    prompt = PromptTemplate.from_template(
        GROUNDING_RULES +
        """You are an exam evaluator.
Based STRICTLY on the provided CONTEXT, generate a multiple-choice quiz
(5 questions) for the topic "{topic}".

ABSOLUTE RULES:
- Every question MUST be answerable from the CONTEXT alone.
- Do NOT create questions about topics not in the CONTEXT.
- Each question must have exactly 4 options labeled A, B, C, D.
- Provide the correct answer letter, an explanation, and a source citation.

Output STRICTLY as a JSON array. Each element must have these exact keys:
  "question", "options" (object with keys A/B/C/D), "correct", "explanation", "citation"

Example:
[
  {{
    "question": "What is the primary key in a relational database?",
    "options": {{
      "A": "A foreign key reference",
      "B": "A unique identifier for each row",
      "C": "An index on all columns",
      "D": "A stored procedure"
    }},
    "correct": "B",
    "explanation": "A primary key uniquely identifies each record in a table.",
    "citation": "Source: database_notes.pdf, Page 3"
  }}
]

CONTEXT:
{context}
"""
    )

    chain = prompt | llm | JsonOutputParser()

    try:
        result = chain.invoke({"context": context, "topic": topic})
        # Unwrap if LLM wrapped inside a dict
        if isinstance(result, dict):
            if "error" in result:
                return result
            for val in result.values():
                if isinstance(val, list):
                    return val
        return result
    except Exception as e:
        return {"error": f"Failed to generate quiz: {str(e)}"}
