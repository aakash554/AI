import streamlit as st
import json
import os
from core_logic import (
    save_uploaded_file,
    ingest_pdf,
    generate_flashcards,
    generate_revision_planner,
    generate_grounded_quiz,
    get_vectorstore,
    clear_knowledge_base,
)


# Cached wrappers (reset when new PDFs are ingested)


@st.cache_data(show_spinner=False)
def cached_generate_flashcards(topic: str):
    return generate_flashcards(topic)

@st.cache_data(show_spinner=False)
def cached_generate_revision_planner(topic: str):
    return generate_revision_planner(topic)

@st.cache_data(show_spinner=False)
def cached_generate_grounded_quiz(topic: str):
    return generate_grounded_quiz(topic)


# Page config


st.set_page_config(page_title="Exam Revision Assistant", layout="wide")

st.title("   Exam Revision Assistant")
st.markdown(
    "Your AI study companion powered by **Gemini 2.5 Flash**, "
    "**LangChain**, **ChromaDB**, and **Hybrid Retrieval** (Semantic + BM25)."
)


# Sidebar


with st.sidebar:
    st.header("Upload Study Materials")
    uploaded_files = st.file_uploader(
        "Upload PDF files", type=["pdf"], accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("Process PDFs", use_container_width=True)
    with col2:
        clear_btn = st.button(" Clear KB", use_container_width=True)

    if clear_btn:
        clear_knowledge_base()
        st.cache_data.clear()
        st.success("Knowledge base cleared! Upload new PDFs to start fresh.")

    if process_btn:
        if uploaded_files:
            with st.spinner("Ingesting documents..."):
                total_chunks = 0
                for file in uploaded_files:
                    path = save_uploaded_file(file)
                    chunks = ingest_pdf(path)
                    total_chunks += chunks
                st.cache_data.clear()
                st.success(
                    f" Processed {len(uploaded_files)} file(s) → {total_chunks} chunks embedded!"
                )
        else:
            st.warning("Please upload at least one PDF first.")

    st.divider()
    st.info(
        " **How it works:** PDFs are chunked & embedded into ChromaDB. "
        "Queries use **hybrid retrieval** (semantic + keyword search) so the "
        "model answers *only* from your documents — no hallucination."
    )


# Main interface


st.header(" Choose an Assistant Tool")

tool_choice = st.radio(
    "Select a task:",
    ("Flashcard Generator", "Revision Planner", "Grounded Quiz"),
    horizontal=True,
)

topic = st.text_input("Enter the topic or subject you want to focus on:")

if st.button("Generate"):
    #  Guard: API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error(" GOOGLE_API_KEY not found. Please set it in the `.env` file.")
    # Guard: topic 
    elif not topic:
        st.error("Please enter a topic.")
    # Guard: knowledge base must exist 
    elif not get_vectorstore():
        st.error(
            " No documents in the knowledge base. "
            "Please upload and process PDFs first — the model will **only** "
            "answer from your uploaded documents."
        )
    else:
        if tool_choice == "Revision Planner":
            with st.spinner("Structuring your study schedule..."):
                plan_result = cached_generate_revision_planner(topic)
                st.subheader(" Study Plan")
                st.markdown(plan_result)

        elif tool_choice == "Flashcard Generator":
            with st.spinner("Analyzing texts and generating flashcards..."):
                st.subheader(" Your Flashcards")
                flashcards_result = cached_generate_flashcards(topic)

                # Fix: unwrap if LLM wrapped the list inside a dict
                if isinstance(flashcards_result, dict) and "error" not in flashcards_result:
                    for val in flashcards_result.values():
                        if isinstance(val, list):
                            flashcards_result = val
                            break

                if isinstance(flashcards_result, dict) and "error" in flashcards_result:
                    st.error(flashcards_result["error"])
                elif not isinstance(flashcards_result, list):
                    st.error("The LLM output was malformed. Please try again.")
                else:
                    for i, card in enumerate(flashcards_result):
                        if isinstance(card, dict):
                            with st.expander(
                                f"**Term {i+1}:** {card.get('term', 'Unknown')}"
                            ):
                                st.write(card.get("definition", "No definition."))

                    json_str = json.dumps(flashcards_result, indent=4)
                    st.download_button(
                        label="Download Flashcards (JSON)",
                        data=json_str,
                        file_name=f"{topic.replace(' ', '_')}_flashcards.json",
                        mime="application/json",
                        use_container_width=True,
                    )

        elif tool_choice == "Grounded Quiz":
            with st.spinner("Drafting questions and citing sources..."):
                quiz_result = cached_generate_grounded_quiz(topic)

                # Handle errors / not available
                if isinstance(quiz_result, dict) and "error" in quiz_result:
                    st.error(quiz_result["error"])
                elif not isinstance(quiz_result, list) or len(quiz_result) == 0:
                    st.error("The quiz could not be generated. Please try again.")
                else:
                    st.session_state["quiz_data"] = quiz_result
                    st.session_state["quiz_submitted"] = False


# Interactive quiz renderer (persists via session_state)


if "quiz_data" in st.session_state and st.session_state["quiz_data"]:
    quiz = st.session_state["quiz_data"]
    st.subheader(" Grounded Quiz")
    st.markdown(f"**{len(quiz)} questions** — select your answers then click **Submit Quiz**.")

    user_answers = {}
    for i, q in enumerate(quiz):
        if not isinstance(q, dict):
            continue
        st.markdown(f"---")
        st.markdown(f"**Q{i+1}.** {q.get('question', '')}")
        options = q.get("options", {})
        choices = [f"{k}. {v}" for k, v in options.items()]
        selected = st.radio(
            f"Your answer for Q{i+1}:",
            choices,
            key=f"quiz_q_{i}",
            index=None,
            label_visibility="collapsed",
        )
        if selected:
            user_answers[i] = selected[0]  

    st.markdown("---")
    if st.button(" Submit Quiz", use_container_width=True):
        st.session_state["quiz_submitted"] = True

    if st.session_state.get("quiz_submitted"):
        score = 0
        total = len(quiz)

        for i, q in enumerate(quiz):
            if not isinstance(q, dict):
                continue
            correct = q.get("correct", "").strip().upper()
            user = user_answers.get(i, "")
            is_correct = user == correct

            if is_correct:
                score += 1
                st.success(f"**Q{i+1}:  Correct!**")
            elif user:
                st.error(
                    f"**Q{i+1}:  Wrong** — You chose **{user}**, "
                    f"correct answer is **{correct}**"
                )
            else:
                st.warning(f"**Q{i+1}:  Not answered** — Correct: **{correct}**")

            with st.expander(f" Explanation for Q{i+1}"):
                st.write(q.get("explanation", "No explanation provided."))
                st.caption(q.get("citation", ""))

        st.markdown("---")
        pct = int((score / total) * 100) if total else 0
        if pct >= 80:
            st.success(f"## Score: {score}/{total} ({pct}%) — Excellent!")
        elif pct >= 50:
            st.info(f"## Score: {score}/{total} ({pct}%) — Good effort!")
        else:
            st.warning(f"## Score: {score}/{total} ({pct}%) — Keep revising!")
