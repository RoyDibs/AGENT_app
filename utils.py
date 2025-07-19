from langchain.tools import tool

@tool
def greeting_tool(query: str):
    """Use this tool to respond to greetings like 'hi', 'hello', or 'hey'."""
    return "My name is Aria, what can I help you?"
