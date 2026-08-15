from agents import Agent , Runner,function_tool,input_guardrail,GuardrailFunctionOutput,trace,InputGuardrailTripwireTriggered
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd
import os
import asyncio


load_dotenv()
model=os.getenv("OPENAI_MODEL", "gpt-4o-mini")

os.makedirs("output",exist_ok=True)
@function_tool
def csv_parser(filename:str)->dict:  
    """Parse a CSV file and return its structure: columns, types, shape, and missing values."""
    try:
        data =pd.read_csv(filename)
        column_name = data.columns
        Type=data.dtypes
        shape=data.shape
        missing_values=data.isna().sum()
        return {
            "column_name":column_name.to_list(),
            "Type":Type.to_dict(),
            "shape":f"rows:{shape[0]} columns:{shape[1]}",
            "missing_values":missing_values.to_dict()
        }
    except Exception as e:
        return f"Error: {e}"
@function_tool
def pattern_finder(filename:str)->dict:
    """Find statistical patterns in a CSV: correlation, min/max values."""
    try:
        data=pd.read_csv(filename)
        numerical_columns=data.select_dtypes(include=['number'])
        correlattion=numerical_columns.corr().to_dict()
        high_value=numerical_columns.max().to_dict()
        low_value=numerical_columns.min().to_dict()

        return {
            "high_value":high_value,
            "low_value":low_value,
            "correlattion":correlattion,
        
        }
    except Exception as e:
        return f"Error: {e}"

@input_guardrail
def end_agent(ctx,input:str):
     exit_requested = "exit" in input.lower()
     return GuardrailFunctionOutput(output_info={"exit":True},tripwire_triggered=exit_requested)

@function_tool
def read(filename:str):
    """Read and return full CSV content as a string."""
    try:
        return f"{pd.read_csv(filename).to_string()}"
    except Exception as e:
        return f"Error reading file: {e}"
QA_Agent=Agent(name="CSV Assistant",instructions="You are a helpful assistant that can answer questions about the data in the CSV file.you can use the read tool that will helps to read the csv file ",tools=[read],input_guardrails=[end_agent],model=model)
chat_assist=QA_Agent.as_tool(tool_name="chat_assisstant",
tool_description="You are a helpful assistant that can answer questions about the data in the CSV file")

@function_tool
def graph_generated(csv_file)->str:
    """Generate bar charts for all columns (categorical and numerical) and save them to /output."""
    data=pd.read_csv(csv_file)
    text_columns=data.select_dtypes(include=['object'])
    numerical_columns=data.select_dtypes(include=['number'])
    for column in text_columns:
        data[column].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {column}")
        plt.savefig(f"output/graph_{column}.png")
        plt.close()
    for column in numerical_columns:
        data[column].plot(kind='bar')
        plt.title(f"Distribution of {column}")
        plt.savefig(f"output/graph_{column}.png")
        plt.close()
    return "Graphs generated successfully"
@function_tool
def write_report(content: str) -> str:
    """
    Save the final markdown analysis report to output/report.md.
    The content should be a complete markdown string.
    """
    try:
        path = "output/report.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Report saved successfully to {path}"
    except Exception as e:
        return f"Error saving report: {e}"
analyse_agent=Agent(name="Data Analyst",
instructions="""You are an advanced Data Analyst agent.

Your job is to fully analyze a dataset and produce a complete analysis report.

You must strictly use the available tools in the following sequence:
1.csv_parser
2.pattern_finder
3.chat_assist
4.graph_generated
5.write_report 
Report Writer (Final Step)
   Create a complete markdown report called report.md  using the write report tool that includes:

   - Dataset Overview
   - Key Patterns and Findings
   - Analytical Questions and Answers
   - Visualizations

   The report must:
   - Combine all findings from previous steps
   - Embed the generated charts using markdown image syntax
   - Include a section called "Key Insights"
   - Clearly explain the most important conclusions from the analysis
   - Save the final report as report.md

Important Rules:
- Always follow the tool sequence exactly.
- Do not skip any tool.
- Use insights from earlier steps to inform later steps.
- The final output must be a clear, well-structured analysis report.
- the user will only provide the path for csv file the should be passed to tools as well
""",
model=model,
tools=[csv_parser,pattern_finder,graph_generated,read,chat_assist,write_report])

async def main():
    
    print(f"🚀 Starting analysis on:\n")
    with trace("hi"):
        try:
            result = await Runner.run(analyse_agent, "\csv_file\medical_insurance_2026_kaggle.csv")
            print("\n===== FINAL OUTPUT =====")
            print(result.final_output)
            print("\n✅ Done! Check the output/ folder for graphs and report.md")
        except InputGuardrailTripwireTriggered as e:
            print("\n⚠️ Session ended — user requested exit.")
            print(f"Guardrail info: {e.guardrail_result.output.output_info}")

if __name__ == "__main__":
    asyncio.run(main())

