import asyncio
import os
from  dotenv  import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import  Agent,trace,Runner,OpenAIChatCompletionsModel
from agents.mcp import MCPServerStdio
import functools
import subprocess
import agents.mcp.server

load_dotenv()
openrouter_api_key = os.getenv('API_KEY')
openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
model=OpenAIChatCompletionsModel(model="openai/gpt-oss-20b:free",openai_client=openrouter_client)
agents.mcp.server.stdio_client=functools.partial(agents.mcp.server.stdio_client,errlog=subprocess.DEVNULL)
report_path=os.path.abspath(os.path.join(os.getcwd(),"report"))
os.makedirs(report_path,exist_ok=True)
files_params={"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem",report_path]}
fetch_params={"command": "npx","args": [ "@playwright/mcp@latest"]}
custom_params={"command":"uv","args":["run","-m","notes"]}
async def main():
    async with MCPServerStdio(name="filesystem",params=files_params,client_session_timeout_seconds=120) as file:
        async with MCPServerStdio(name="fetch",params=fetch_params,client_session_timeout_seconds=180) as fetch:
            async with MCPServerStdio(name="notes",params=custom_params,client_session_timeout_seconds=120) as custom:
                agent=Agent(name="Research Assistant",
                instructions="You are a helpful research assistant. "
                    "You can search the web, manage research notes, and read/write files. "
                    "When asked to research, save important findings to the notes tool. "
                    "When asked to create or store a report, save it in the report folder.",
                    model=model,
                    mcp_servers=[file,fetch,custom]
                )

                with trace("researcher"):
                    ins= input("Enter the specific area: ")
                    result = Runner.run_streamed(
                        agent,
                        f"research about {ins}"
                    )

                async for event in result.stream_events():
                    if event.type == "raw_response_event" and  isinstance(event.data, ResponseTextDeltaEvent):
                            print(event.data.delta, end="", flush=True)
if __name__=="__main__":
    asyncio.run(main())