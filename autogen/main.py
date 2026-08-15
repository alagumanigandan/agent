import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import MessageContext,RoutedAgent,AgentId,message_handler
from autogen_core import SingleThreadedAgentRuntime
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dataclasses import dataclass,field
import os
from dotenv import load_dotenv
from typing import List,Optional,Literal
from pydantic import BaseModel


load_dotenv()
model=os.getenv("MODEL")
import subprocess
import os

SANDBOX_DIR = os.path.abspath("output")


def run_sandbox_python(filename: str) -> str:
    """
    Execute a Python file inside an ephemeral Docker container.

    The output directory is mounted into the container and the
    generated Python file is executed using uv.
    """

    file_path = os.path.join(SANDBOX_DIR, filename)

    if not os.path.isfile(file_path):
        return f"ERROR: File not found: {filename}"

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "256m",
                "--cpus", "1",
                "-v", f"{SANDBOX_DIR}:/workspace",
                "-w", "/workspace",
                "ghcr.io/astral-sh/uv:python3.13-bookworm-slim",
                "uv", "run", filename,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return (
                f"Execution failed.\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        return result.stdout

    except subprocess.TimeoutExpired:
        return "ERROR: Code execution timed out."

    except Exception as e:
        return f"ERROR: {str(e)}"
@dataclass
class Message:
    count: int
    message: str
    subtasks: List[str]=field(default_factory=list)
    code: Optional[str] = None
    feedback: Optional[str] = None
class Next_agent(BaseModel):
    handoffs : Literal["Coder","Documentor"]
    feedback: str
class DocumentationResult(BaseModel):
    documentation: str
class Planner(RoutedAgent):
    system_message = """
        You are a Planner Agent.
        Your only job is to break down a goal into task cards.
        Return tasks only. No markdown. No explanation. No extra text.
        """
    def __init__(self) -> None:
        super().__init__("Planner")
       
        self.model=OpenAIChatCompletionClient(
        model=model)
        self._delegate=AssistantAgent(
            name="Planner",
            model_client=self.model,
            system_message=self.system_message,
            
        )
    @message_handler
    async def planner(self,message:Message,ctx:MessageContext)->Message:
            text=TextMessage(content=f"{message}",source="user")
            result=await self._delegate.on_messages([text],ctx.cancellation_token)
            tasks = result.chat_message.content.split("\n")
            tasks = [t.strip() for t in tasks if t.strip()]
            return Message(count=message.count+1,message=message.message,subtasks=tasks)

class Coder(RoutedAgent):
    system_message="""You are a Coder Agent.
                        You receive list  task card and write the code for it using the autogen framework.
                        Return the code only. No explanation. No comments. No extra text.
                        Just the raw code string ready to be executed.
                        - Write only the code. Nothing else.
                        - Do not explain what you wrote.
                        - Do not add comments inside the code.
                        - Do not add anything before or after the code.
                        - Write only what the task card asks for. Nothing more.
                        - Your code will be tested against the success criteria in the task card.
                        - Write testable code — no hardcoded values, no global side effects."""
        
    def __init__(self) ->None:
         super().__init__("Coder")
         self.model=OpenAIChatCompletionClient(
                    model=model,
                    model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "structured_output": False,
        "family": "unknown"
    }) 
         self._delegate=AssistantAgent(
            name="Coder",
            model_client=self.model,
            system_message=self.system_message
        
        )
    @message_handler
    async def handle_message_on(self,message:Message,ctx:MessageContext)->Message:
        print("Coder writing code...")
        text=text = TextMessage(
    content=f"""
    Task:
    {message.message}

    Tasks:
    {message.subtasks}

    Previous code:
    {message.code}

    Tester feedback:
    {message.feedback}

    If tester feedback exists, fix the previous code according to the feedback.
    If this is the first attempt, generate the implementation from the task cards.

    Return only the complete Python code.
    """,
        source="user"
    )
        respond= await self._delegate.on_messages([text],cancellation_token=ctx.cancellation_token)
        os.makedirs("output", exist_ok=True)
        with open(f"output/agent{message.count}.py",mode="w",encoding="utf-8") as f :
            f.write(respond.chat_message.content)
        return Message(message=message.message, subtasks=message.subtasks, code=respond.chat_message.content,count=message.count+1)
class Tester(RoutedAgent):

    system_message = """
You are a Tester Agent.

Your job is to test the generated Python code.

You MUST use the run_sandbox_python tool to execute the generated file.

After execution:
- If execution succeeds and the code satisfies the task, set handoffs to "Documentor".
- If execution fails or the code does not satisfy the task, set handoffs to "Coder".
- Put the exact error and required fix in feedback.

Do not rewrite the code yourself.
Do not assume the code works without executing it.
"""

    def __init__(self) -> None:

        super().__init__("Tester")

        self.model = OpenAIChatCompletionClient(
            model=model,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "structured_output": True,
                "family": "unknown"
            }
        )

        self._delegate = AssistantAgent(
            name="Tester",
            model_client=self.model,
            system_message=self.system_message,
            output_content_type=Next_agent,
            tools=[run_sandbox_python]
        )

    @message_handler
    async def handle_message_on(
        self,
        message: Message,
        ctx: MessageContext
    ) -> Message:

        filename = f"agent{message.count - 1}.py"

        text = TextMessage(
            content=f"""
Test this generated Python code.

Filename:
{filename}

Code:
{message.code}

Use the run_sandbox_python tool to execute:
{filename}

Analyze the execution result.

If it works:
handoffs = Documentor

If it fails:
handoffs = Coder

Put the reason and required fix in feedback.
""",
            source="user"
        )

        respond = await self._delegate.on_messages(
            [text],
            ctx.cancellation_token
        )

        return Message(
            message=respond.chat_message.content.handoffs,
            subtasks=message.subtasks,
            code=message.code,
            count=message.count,
            feedback=respond.chat_message.content.feedback
        )
class Documentor(RoutedAgent):

    system_prompt = """
You are a Documentation Agent.

Your job is to explain the generated code clearly.

Rules:
- Explain what the code does
- Explain the main logic
- Explain inputs and outputs
- Write clean technical documentation
- Do not rewrite the code
"""

    def __init__(self) -> None:

        super().__init__("Documentor")

        self.model = OpenAIChatCompletionClient(
            model=model,
            model_info={
        "vision": False,
        "function_calling": False,
        "json_output": False,
        "structured_output": True,
        "family": "unknown"
    }
        )

        self._delegate = AssistantAgent(
            name="Documentor",
            model_client=self.model,
            system_message=self.system_prompt,
            output_content_type=DocumentationResult
        )

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext)->Message:

        text = TextMessage(
            content=f"Write documentation for this code:\n{message.code}",
            source="user"
        )

        respond = await self._delegate.on_messages(
            [text],
            ctx.cancellation_token
        )

 

        with open(f"output/documentation{message.count}.md", "w", encoding="utf-8") as f:
            f.write(respond.chat_message.content.documentation)

        return Message(
        message=respond.chat_message.content.handoffs,  
        subtasks=message.subtasks,
        code=message.code,
        count=message.count
    )
        
async def main():
    runtime=SingleThreadedAgentRuntime()
    await Planner.register(runtime,"planner_agent",lambda:Planner())
    await Coder.register(runtime,"Coder",lambda:Coder())
    await Tester.register(runtime,"Tester",lambda:Tester())
    await Documentor.register(runtime,"Documentor",lambda:Documentor())
    
    Planer_id=AgentId("planner_agent","default")
    Coder_id=AgentId("Coder","default")
    Tester_id=AgentId("Tester","default")
    Documentor_id=AgentId("Documentor","default")
    await runtime.start( )
    
    plan = await runtime.send_message(Message(message="Build an agent that analyzes gold rate and outputs whether to buy or not",count=0),Planer_id)

    code = await runtime.send_message(plan, Coder_id)

    test_result=await runtime.send_message(code, Tester_id)
    while test_result.message != "Documentor":

        
        print("Feedback:", test_result.feedback)

        code = await runtime.send_message(test_result,Coder_id)
        test_result = await runtime.send_message(code,Tester_id)
    documentation = await runtime.send_message( test_result, Documentor_id)

    await runtime.stop()
if __name__=="__main__":
    asyncio.run(main())