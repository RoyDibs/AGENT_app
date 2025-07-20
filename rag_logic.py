import os
from pathlib import Path
from typing import TypedDict, List, Optional, Literal

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
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
        "You are an AI assistant tasked with selecting the most relevant document "
        "from the following list based on the user's question.\n\n"
        f"Available documents:\n{chr(10).join(list(pdf_library.keys()))}\n\n"
        f"User's question:\n{question}\n\n"
        "Which document best matches the user's question? "
        "Respond with the exact document name (not filename)."
        "if reselect the document from rewrite_question, please choose another document to answer."
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

    # inline generation here like in CLI
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer from the context provided, just say that you don't know. "
        "Keep the answer concise and directly address the question.\n\n"
        "CONTEXT:\n{context}\n\n"
        "USER QUESTION: {question}"
    )
    prompt = system_prompt.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"context": context, "messages": [response]}


class GradeDocuments(BaseModel):
    binary_score: Literal['yes', 'no'] = Field(...)

def grade_documents(state: GraphState) -> Literal["generate_answer", "rewrite_question"]:
    prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )
    grading_prompt = prompt.format(question=state["question"], context=state["context"])
    grader = llm.with_structured_output(GradeDocuments)
    response = grader.invoke(grading_prompt)
    return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

def rewrite_question(state: GraphState) -> dict:
    question = state["question"]
    prompt = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:\n"
        "-------\n"
        f"{question}\n"
        "-------\n"
        "Formulate an improved question:"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"question": response.content}

def generate_answer(state: GraphState) -> dict:
    question = state["question"]
    context = state["context"]

    gen_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Keep the answer concise and use at most three sentences. "
    "When expressing mathematical results, write them in clear, human-readable math format, using symbols like ∫, ∑, π, √, and fractions like 1/2 instead of LaTeX."
        
    "Question: {question} \n"
    "Context: {context}"
    )

    user_prompt = (
        f"Question: {question}\n"
        f"Context: {context}"
    )

    response = llm.invoke([
        SystemMessage(content=gen_prompt),
        HumanMessage(content=user_prompt)
    ])
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
