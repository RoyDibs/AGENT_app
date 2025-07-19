import os

# Environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Document paths
project_root = os.path.dirname(os.path.abspath(__file__))
pdf_library = {
    "chapter1": os.path.join(project_root, "chapter1.docx"),
    "chapter2": os.path.join(project_root, "chapter2.docx"),
    "chapter3": os.path.join(project_root, "chapter3.docx"),
    "course information": os.path.join(project_root, "course syllabus.docx"),
}

persist_directory = os.path.join(os.path.expanduser("~"), "aria_vectorstore")
os.makedirs(persist_directory, exist_ok=True)
