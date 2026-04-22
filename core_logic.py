import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

DB_DIR = "./exam_knowledge_db"
TEMP_DIR = "./temp_uploads"

LLM_MODEL = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

def init_llm(json_mode=False):
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    if json_mode:
        llm = llm.bind(format="json")
    return llm

def init_embeddings():
    return OllamaEmbeddings(model=EMBED_MODEL)

def save_uploaded_file(uploaded_file):
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Metadata normalization
    for doc in docs:
        doc.metadata['source_file'] = os.path.basename(file_path)
        # default pypdf loader provides 'page' metadata
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    embeddings = init_embeddings()
    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
    
    return len(splits)

def get_vectorstore():
    embeddings = init_embeddings()
    if not os.path.exists(DB_DIR):
        return None
    try:
        # Chroma handles loading from directory
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    except Exception:
        return None

def get_retriever():
    vs = get_vectorstore()
    if vs:
        return vs.as_retriever(search_kwargs={"k": 5})
    return None

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source_file')} - Page {doc.metadata.get('page', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

def generate_flashcards(topic: str):
    retriever = get_retriever()
    if not retriever:
        return {"error": "Knowledge base is empty. Please upload some PDFs first."}

    llm = init_llm(json_mode=True)
    
    prompt = PromptTemplate.from_template(
        """You are an exam revision assistant. Based entirely on the following context, extract key concepts and generate a list of flashcards for the topic: "{topic}".
        Format the output strictly as a JSON array of objects. 
        Each object MUST have exactly two formatting keys: "term" and "definition".
        
        Example Output Format:
        [
            {{"term": "Database", "definition": "An organized collection of data."}},
            {{"term": "SQL", "definition": "Structured Query Language."}}
        ]
        
        Context: {context}
        """
    )
    
    chain = (
        {"context": retriever | format_docs, "topic": RunnablePassthrough()}
        | prompt
        | llm
        | JsonOutputParser()
    )
    
    try:
        return chain.invoke(topic)
    except Exception as e:
        return {"error": f"Failed to generate flashcards: {str(e)}"}

def generate_revision_planner(topic: str):
    retriever = get_retriever()
    if not retriever:
        return "Knowledge base is empty. Please upload some PDFs first."

    llm = init_llm()
    
    prompt = PromptTemplate.from_template(
        """You are a revision planner assistant. Based on the following context, create a structured Markdown study schedule for the given topic: "{topic}".
        If the context includes weightage or importance for specific sub-topics, use that to prioritize the schedule. 
        Split the revision into logical days or sessions.
        
        Context: {context}
        
        Markdown Schedule:"""
    )
    
    chain = (
        {"context": retriever | format_docs, "topic": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(topic)

def generate_grounded_quiz(topic: str):
    retriever = get_retriever()
    if not retriever:
        return "Knowledge base is empty. Please upload some PDFs first."

    llm = init_llm()
    
    prompt = PromptTemplate.from_template(
        """You are an exam evaluator. Based strictly on the provided context, generate a multiple-choice quiz (3-5 questions) for the topic "{topic}".
        For every question, you MUST provide the correct answer and strictly include a citation pointing to the specific "Source" and "Page" provided in the context.
        
        Do NOT use outside knowledge. If the context does not contain enough information, state that clearly.
        
        Context: {context}
        
        Quiz Output:"""
    )
    
    chain = (
        {"context": retriever | format_docs, "topic": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(topic)
