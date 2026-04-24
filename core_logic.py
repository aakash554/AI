import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

DB_DIR = "./exam_knowledge_db_gemini"
TEMP_DIR = "./temp_uploads"

LLM_MODEL = "models/gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"

def init_llm(json_mode=False):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    return llm

def init_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

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
        return vs.as_retriever(search_kwargs={"k": 10}) # Fetch 10 chunks to avoid missing important details
    return None

def format_docs(docs):
    return "\n\n".join(f"Source: {doc.metadata.get('source_file')} - Page {doc.metadata.get('page', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

def generate_flashcards(topic: str):
    retriever = get_retriever()
    if not retriever:
        return {"error": "Knowledge base is empty. Please upload some PDFs first."}

    llm = init_llm(json_mode=True)
    
    prompt = PromptTemplate.from_template(
        """You are a strict exam revision assistant. Your task is to extract highly specific key concepts and generate a detailed list of flashcards for the topic: "{topic}".
        You must rely SOLELY on the provided context. Do not use outside knowledge. 
        Ensure that the definitions capture the precise nuances, terminology, and important details mentioned in the text instead of providing general summaries.
        
        Format the output strictly as a JSON array of objects. 
        Each object MUST have exactly two formatting keys: "term" and "definition".
        
        Example Output Format:
        [
            {{"term": "Specific Term from Text", "definition": "Precise, detailed explanation exactly as described in the context."}}
        ]
        
        Context:
        {context}
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
        """You are a strict revision planner assistant. Based ONLY on the following context, create a highly detailed, structured Markdown study schedule for the given topic: "{topic}".
        Incorporate specific sub-topics, exact definitions, and important nuances directly from the text into the schedule so the user knows exactly what to study.
        If the context includes weightage or importance for specific sub-topics, use that to prioritize the schedule. 
        Split the revision into logical days or sessions, and list the specific bullet points to cover under each session.
        Do not use outside knowledge to fill in gaps.
        
        Context:
        {context}
        
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
        """You are a strict exam evaluator. Based ONLY on the provided context, generate a detailed multiple-choice quiz (3-5 questions) for the topic "{topic}".
        The questions should test highly specific, detailed knowledge and facts from the text, not just high-level or general concepts.
        For every question, you MUST provide the correct answer and strictly include a precise citation pointing to the specific "Source" and "Page" provided in the context.
        
        Do NOT use outside knowledge. If the context does not contain enough information to generate questions, state that clearly.
        
        Context:
        {context}
        
        Quiz Output:"""
    )
    
    chain = (
        {"context": retriever | format_docs, "topic": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke(topic)
