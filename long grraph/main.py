from pydantic import BaseModel
from langgraph.graph import START,END,StateGraph  
from langchain_openai import ChatOpenAI
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import GoogleSerperRun
from langchain.tools import tool
import os 
from langchain_community.utilities import GoogleSerperAPIWrapper

import uuid
from dotenv import load_dotenv
# import gradio as gr

load_dotenv()
model=os.getenv("MODEL")
class Research(TypedDict):
    message:Annotated[list,add_messages]
    topic: str
    subtopics: list[str]
    findings: dict
    knowledge: str
    report: str
    current_index: int
    
graph_builder=StateGraph(Research)
llm=ChatOpenAI(model=model)
class Subtopics(BaseModel):
    subtopics: list[str]
structure_llm=llm.with_structured_output(Subtopics)
def topic_Analyser(state:Research)-> Research:
    res=structure_llm.invoke(f"Break this topic into  clear subtopics: {state['topic']}")
    return{"subtopics":res.subtopics,"current_index":0}
def Task_Planner(state:Research)->Research:
    res=structure_llm.invoke(f"Given these subtopics: {state['subtopics']}. "
        f"Prioritize them in order of importance for research.")
    return{"subtopics":res.subtopics,"current_index":0}
serper_search=GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper())

@tool
def save_key_fact(subtopic: str, fact: str) -> str:
    """Save a single standout fact about a subtopic to a running notes file,
    separate from the main research summary. Use this when you find something
    especially important or surprising that deserves to be flagged on its own."""
    with open("output/key_facts.md", "a", encoding="utf-8") as f:
        f.write(f"- **{subtopic}**: {fact}\n")
    return f"Saved key fact for {subtopic}"
def Research_node(state:Research)->Research:
    if not state.get("subtopics"):
        return {"findings": {}, "current_index": 0}
    current = state["subtopics"][state["current_index"]]
    history = state.get("message", [])
    last = history[-1] if history else None
    coming_from_tool = last is not None and getattr(last, "type", None) == "tool"

    if coming_from_tool:
        res = llm_with_tools.invoke(history)
    else:
        human_message = HumanMessage(
            content=f"""You are a research expert.
                Research this specific subtopic: {current}
                Use web search to find relevant information.
                If you find one especially important or surprising fact,
                save it with save_key_fact before writing your summary.
                Provide a detailed summary with key findings.
                Keep it factual and concise.""")
        res = llm_with_tools.invoke([human_message])

    if getattr(res, "tool_calls", None):
        if coming_from_tool:
            return {"message": [res]}
        return {"message": [human_message, res]}
    finding = state.get("findings", {})
    finding[current] = res.content
    result = {
        "findings": finding,
        "current_index": state["current_index"] + 1,
        "message": [res],
    }
    if not coming_from_tool:
        result["message"] = [human_message, res]
    return result

def  Knowledge_Builder(state:Research)->Research:
    findings = state.get("findings", {})
    res = llm.invoke(
        f"""Connect these research findings and find relationships between them:
        {findings}
        Identify key patterns and insights."""
    )
    
    return {"knowledge": res.content}
def decision(state:Research)->Research:
    TOT=len(state["subtopics"])
    if  state["current_index"] < TOT:
        return "continue"
    else:
        return "end"
os.makedirs("output", exist_ok=True) 
def report_generator(state:Research)->Research:
    res=llm.invoke(f"""You are a professional research report writer.

    Topic: {state['topic']}

    Research Findings:
    {state.get('findings', {})}

    Key Connections and Insights:
    {state.get('knowledge', '')}

    Write a complete markdown research report with:
    1. Title
    2. Executive Summary
    3. Key Findings for each subtopic
    4. Connections and Patterns
    5. Final Conclusion

    Make it clear, structured and professional.""")
    with open(f"output/{state['topic']}.md", "w", encoding="utf-8") as f:
        f.write(res.content)
    return{"report":res.content}
thread_id=str(uuid.uuid4())
memory=MemorySaver()
tools=[serper_search,save_key_fact]
llm_with_tools=llm.bind_tools(tools)
graph_builder.add_node("topic_analyzer",topic_Analyser)
graph_builder.add_node("task_planner",Task_Planner)
graph_builder.add_node("researcher",Research_node)
graph_builder.add_node("knowledge_builder",Knowledge_Builder)
graph_builder.add_node("report_generator",report_generator)
graph_builder.add_node("tool",ToolNode(tools))
graph_builder.add_edge(START,"topic_analyzer")
graph_builder.add_edge("topic_analyzer","task_planner")
graph_builder.add_edge("task_planner","researcher")
graph_builder.add_conditional_edges("researcher",tools_condition,{"tools":"tool","__end__":"knowledge_builder"})
graph_builder.add_conditional_edges("knowledge_builder",decision,{"continue":"researcher","end":"report_generator"})
graph_builder.add_edge("tool","researcher")
graph_builder.add_edge("report_generator",END)
config={"configurable":{"thread_id":thread_id}}
graph=graph_builder.compile(checkpointer=memory)

if __name__=="__main__":
    topic=input("Enter research topic: ")
    result=graph.invoke({
        "topic": topic,
        "subtopics": [],
        "findings": {},
        "knowledge": "",
        "report": "",
        "current_index": 0
    },config=config)
    print(result)
