import os
from pathlib import Path
from typing import TypedDict, List, Optional, Literal

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from llama_index.readers.docling import DoclingReader
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool
from pydantic import BaseModel, Field

from config import pdf_library
from utils import greeting_tool

DEBUG = False

llm = init_chat_model("openai:gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools([greeting_tool])

class GraphState(TypedDict):
    question: str
    context: Optional[str]
    messages: List[BaseMessage]
    selected_file: Optional[str]

def tool_check(state: GraphState) -> dict:
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

def select_pdf_from_question(state: GraphState) -> dict:
    question = state["question"]
    pdf_prompt = (
        f"Available documents:\n{chr(10).join(pdf_library.keys())}\n\n"
        f"User's question:\n{question}\n\n"
        "Which document best matches the user's question? Respond with exact document name."
    )
    response = llm.invoke(pdf_prompt)
    selected_name = response.content.strip().lower()
    selected_file = pdf_library.get(selected_name) or list(pdf_library.values())[0]
    return {"selected_file": selected_file}

def retrieve_and_answer_node(state: GraphState) -> dict:
    question = state["question"]
    selected_file = state["selected_file"]
    reader = DoclingReader()
    docs = reader.load_data(str(Path(selected_file).resolve()))
    doc_splits = [Document(page_content=node.text, metadata=node.metadata) for node in docs]

    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(doc_splits, embedding)
    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    retrieved_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    return {"context": context}

class GradeDocuments(BaseModel):
    binary_score: Literal['yes', 'no'] = Field(...)

def grade_documents(state: GraphState) -> Literal["generate_answer", "rewrite_question"]:
    prompt = (
        f"Document:\n\n{state['context']}\n\n"
        f"User question:\n{state['question']}\n\n"
        "Is the document relevant? Respond with 'yes' or 'no'."
    )
    grader = llm.with_structured_output(GradeDocuments)
    response = grader.invoke(prompt)
    return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

def rewrite_question(state: GraphState) -> dict:
    prompt = (
        "Rewrite the following question to be clearer:\n"
        f"{state['question']}"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"question": response.content}

def generate_answer(state: GraphState) -> dict:
    prompt = (
        f"Use the context below to answer the question.\n"
        f"Context:\n{state['context']}\n\n"
        f"Question:\n{state['question']}"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# Graph assembly
graph = StateGraph(GraphState)
graph.add_node("tool_check", tool_check)
graph.add_node("tools", ToolNode([greeting_tool]))
graph.add_node("select_pdf", select_pdf_from_question)
graph.add_node("retrieve_and_answer_node", retrieve_and_answer_node)
graph.add_node("rewrite_question", rewrite_question)
graph.add_node("generate_answer", generate_answer)

graph.add_edge(START, "tool_check")
graph.add_conditional_edges("tool_check", tools_condition, {
    "tools": "tools",
    END: "select_pdf"
})
graph.add_edge("tools", END)
graph.add_edge("select_pdf", "retrieve_and_answer_node")
graph.add_conditional_edges("retrieve_and_answer_node", grade_documents)
graph.add_edge("generate_answer", END)
graph.add_edge("rewrite_question", "select_pdf")

rag_agent = graph.compile()
