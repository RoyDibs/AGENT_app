# ARIA Chatbot

ARIA is a document-aware chatbot built using LangGraph, LangChain, and Streamlit. It intelligently selects the relevant course document (from multiple `.docx` files) and answers your questions using OpenAI's GPT-4o.

## Features

- 🔍 Multi-PDF document routing
- 🤖 Tool-based greeting handling
- 📄 RAG (retrieval-augmented generation)
- 🧠 Relevance-based question rewriting
- 🧪 Graph logic via LangGraph

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
