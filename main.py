import streamlit as st
from langchain_core.messages import HumanMessage
from rag_logic import rag_agent
from config import pdf_library

st.set_page_config(page_title="ARIA Chatbot", layout="wide")
st.title('ARIA Web App')
st.divider()
st.write('Enter your question related to the course:')
st.divider()

def submit_question():
    """Streamlit callback: handle input and run LangGraph agent"""
    q = st.session_state.get("user_question", "").strip()
    if not q:
        return
    initial_state = {
        "question": q,
        "messages": [HumanMessage(content=q)]
    }
    try:
        final_state = rag_agent.invoke(initial_state)
        ans = final_state["messages"][-1].content
        st.session_state.history.append((q, ans))
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
        st.markdown(f"**ARIA:** {a}", unsafe_allow_html=True)
        st.divider()

# Create dummy files if not present
for doc in pdf_library.values():
    if not os.path.exists(doc):
        with open(doc, 'w') as f:
            f.write(f"This is a dummy file named {doc}.")

running_agent()
