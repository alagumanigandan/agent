import os 
from typing_extensions import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import  FAISS
from langchain_core.tools import tool
from langgraph.graph import START,StateGraph,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
SYS_PROMPT="""You are a document research assistant. Your job is to answer \
questions using ONLY the information available in the user's uploaded document, \
retrieved through the rag_tool.Rules:
- For any question that could be answered from the document, always call \
rag_tool first before answering. Do not answer from your own general \
knowledge if the question is about the document's content.
- Base your answer strictly on what rag_tool returns. Do not add facts, \
figures, or claims that aren't present in the retrieved passages.
- If rag_tool returns no relevant information, say so plainly — for example: \
"I couldn't find anything about that in the document." Do not guess or fill \
the gap with outside knowledge.
- If the retrieved passages only partially answer the question, answer what \
they support and clearly state what's missing, rather than extrapolating.
- Keep answers concise and directly tied to the source material.
- If a question is clearly unrelated to the document, say that it's outside \
the scope of the document rather than attempting to answer it anyway.
"""
load_dotenv()
loader=PyPDFLoader("path.pdf")
api=os.getenv("API_KEY")
data=loader.load()
textsplit=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunk=textsplit.split_documents(data)
embed=OpenAIEmbeddings(
    model="nvidia/llama-nemotron-embed-vl-1b-v2:free",
    api_key=api,
    base_url="https://openrouter.ai/api/v1",
)
vect_store= FAISS.from_documents(chunk,embed)
vect_store.save_local("index")
retriever =vect_store.as_retriever(search_type="similarity", search_kwargs={'k':4})

@tool
def rag_tool(query:str)->str:
    """Search personal documents and return relevant information."""
    docs=retriever.invoke(query)
    return "\n\n".join(
        doc.page_content
        for doc in docs)
class State(TypedDict):
    messages:Annotated[list,add_messages]
tools=[rag_tool]
llm=ChatOpenAI(
    api_key=api,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-oss-20b:free"
).bind_tools(tools)
def chatbot(state:State)->dict :
    system=SystemMessage(SYS_PROMPT)
    message=[system]+ state["messages"]
    res=llm.invoke(message)
    return{"messages":[res]}
memory=MemorySaver()
builder=StateGraph(State)

builder.add_node("chatbot",chatbot)
builder.add_node("tool",ToolNode(tools))
builder.add_edge(START,"chatbot")
builder.add_conditional_edges("chatbot",tools_condition,{"tools":"tool","__end__":END})
builder.add_edge("tool","chatbot")
graph=builder.compile(checkpointer=memory)
if __name__=="__main__":

 config = {"configurable": {"thread_id": "1"}}
 while True:
        user_input = input("Ask a question: ")
        if user_input.lower() in ("exit", "quit"):
            break
        result = graph.invoke({"messages": [("user", user_input)]}, config=config)
        print(result["messages"][-1].content)
