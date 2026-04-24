import streamlit as st
import json
import os
from core_logic import (
    save_uploaded_file, 
    ingest_pdf, 
    generate_flashcards, 
    generate_revision_planner, 
    generate_grounded_quiz,
    get_vectorstore
)

@st.cache_data(show_spinner=False)
def cached_generate_flashcards(topic: str):
    return generate_flashcards(topic)

@st.cache_data(show_spinner=False)
def cached_generate_revision_planner(topic: str):
    return generate_revision_planner(topic)

@st.cache_data(show_spinner=False)
def cached_generate_grounded_quiz(topic: str):
    return generate_grounded_quiz(topic)

st.set_page_config(page_title="Exam Revision Assistant", layout="wide")

st.title("Exam Revision Assistant")
st.markdown("Your study companion powered by **Google Gemini API**, **Langchain**, and **ChromaDB**.")

# Sidebar for API Key and PDF uploads
with st.sidebar:
    st.header("1. Configuration")
    api_key_input = st.text_input("Enter Google Gemini API Key", type="password")
    if api_key_input:
        os.environ["GOOGLE_API_KEY"] = api_key_input

    st.header("2. Upload Study Materials")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if uploaded_files:
            with st.spinner("Ingesting documents..."):
                total_chunks = 0
                for file in uploaded_files:
                    path = save_uploaded_file(file)
                    chunks = ingest_pdf(path)
                    total_chunks += chunks
                st.cache_data.clear() # Reset cache so new knowledge is used
                st.success(f"Processed {len(uploaded_files)} files. Generated {total_chunks} embeddings!")
        else:
            st.warning("Please upload a PDF first.")
            
    st.divider()
    st.info("Ensure you have provided a valid Google Gemini API Key above.")

# Main Interface
if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
    st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to proceed.")
    st.stop()

st.header("3. Choose an Assistant Tool")

tool_choice = st.radio("Select a task:", 
                       ("Flashcard Generator", "Revision Planner", "Grounded Quiz"), 
                       horizontal=True)

topic = st.text_input("Enter the topic or subject you want to focus on:")

if st.button("Generate"):
    if not topic:
        st.error("Please enter a topic.")
    else:
        # Check if DB exists
        if not get_vectorstore():
            st.error("No documents found in knowledge base. Please upload and process PDFs first.")
        else:
            if tool_choice == "Revision Planner":
                with st.spinner("Structuring your study schedule..."):
                    plan_result = cached_generate_revision_planner(topic)
                    st.subheader("Study Plan")
                    st.markdown(plan_result)

            elif tool_choice == "Flashcard Generator":
                with st.spinner("Analyzing texts and generating flashcards..."):
                    st.subheader("Your Flashcards")
                    flashcards_result = cached_generate_flashcards(topic)
                    
                    # Fix: If the LLM wrapped the JSON list inside a dictionary (e.g. {"cards": [...]})
                    if isinstance(flashcards_result, dict) and "error" not in flashcards_result:
                        for val in flashcards_result.values():
                            if isinstance(val, list):
                                flashcards_result = val
                                break

                    # Handle errors or malformed non-list outputs
                    if isinstance(flashcards_result, dict) and "error" in flashcards_result:
                        st.error(flashcards_result["error"])
                    elif not isinstance(flashcards_result, list):
                        st.error("The LLM output was malformed. Please try generating it again.")
                    else:
                        for i, card in enumerate(flashcards_result):
                            if isinstance(card, dict):
                                with st.expander(f"**Term {i+1}:** {card.get('term', 'Unknown Term')}"):
                                    st.write(card.get('definition', 'No definition provided.'))
                                
                        json_str = json.dumps(flashcards_result, indent=4)
                        st.download_button(
                            label="Download Flashcards (JSON)",
                            data=json_str,
                            file_name=f"{topic.replace(' ', '_')}_flashcards.json",
                            mime="application/json",
                            use_container_width=True
                        )

            elif tool_choice == "Grounded Quiz":
                with st.spinner("Drafting questions and citing sources..."):
                    quiz_result = cached_generate_grounded_quiz(topic)
                    st.subheader("🎯 Grounded Quiz")
                    st.markdown(quiz_result)
