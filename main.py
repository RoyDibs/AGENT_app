# Updated main.py to support LaTeX rendering with MathJax in Streamlit

import os
import streamlit as st
from langchain_core.messages import HumanMessage, BaseMessage
from rag_logic import rag_agent
from config import pdf_library

# Streamlit page config
st.set_page_config(page_title="ARIA Chatbot", layout="wide")
st.title('ARIA Web App')
st.divider()
st.write('Enter your question related to the course:')
st.divider()

# Create dummy DOCX files if not present (for dev/testing)
for doc in pdf_library.values():
    if not os.path.exists(doc):
        with open(doc, 'w') as f:
            f.write(f"This is a dummy file named {doc}.")

def submit_question():
    """Streamlit callback: Handle input and run LangGraph agent"""
    q = st.session_state.get("user_question", "").strip()
    if not q:
        return

    initial_state = {
        "question": q,
        "messages": [HumanMessage(content=q)],
    }

    try:
        final_state = rag_agent.invoke(initial_state)
        final_answer = final_state.get("messages", [])[-1]
        response = final_answer.content if hasattr(final_answer, 'content') else final_answer
        st.session_state.history.append((q, response))
    except Exception as e:
        st.session_state.history.append((q, f"❌ Error: {e}"))

    st.session_state.user_question = ""

def running_agent():
    if "history" not in st.session_state:
        st.session_state.history = []

    st.text_input(
        "Please enter your question related to the course:",
        key="user_question",
        on_change=submit_question,
    )

    st.divider()
    for q, a in st.session_state.history[::-1]:
        st.markdown(f"**User:** {q}")

        # Render LaTeX if response contains LaTeX syntax
        if "$" in a or "\\" in a:
            st.markdown("**ARIA:**")
            try:
                st.latex(a.strip().strip("`"))
            except Exception:
                st.markdown(f"**ARIA:** {a}", unsafe_allow_html=True)
        else:
            st.markdown(f"**ARIA:** {a}", unsafe_allow_html=True)

        st.divider()

running_agent()
