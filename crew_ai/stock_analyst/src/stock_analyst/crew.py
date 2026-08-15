from crewai import Agent, Crew, Process, Task


from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List
from pydantic import  BaseModel,Field
from dotenv import load_dotenv
import os

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
load_dotenv()
url=os.getenv("BASE_URL")
API_KEY=os.getenv("API_KEY")

class RiskAssessment(BaseModel):
        risk_level: str=Field(description="Overall risk level of the stock")
        risk_score: int=Field(description="Risk score from 1 (very safe) to 10 (very risky)")
        red_flags: List[str]= Field(description="List of red flags that indicate potential risks")
        key_concerns: List[str]= Field(description="Important risk concerns investors should watch")
        overall_summary: str= Field(description="Short explanation summarizing the risk analysis")
@CrewBase
class StockAnalyst():
    """StockAnalyst crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def Researcher(self)->Agent:
        return Agent(
            config=self.agents_config['Researcher'],
            tools=[SerperDevTool()],
            verbose=True,
            memory=True,
           
        )

    @agent
    def Fundamental_Analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['Fundamental_Analyst'], # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()],
            memory=True,
            
        )
    @agent
    def Sentiment_Analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["Sentiment_Analyst"],
            verbose=True,
            tools=[SerperDevTool()],
           
        )
    @agent
    def Risk_Assessor(self) -> Agent:
        return  Agent(
            config=self.agents_config["Risk_Assessor"],
            verbose=True,
            memory=True,
           
        )
    @agent
    def Portfolio_Reporter(self) -> Agent:
        return Agent(
            config=self.agents_config["Portfolio_Reporter"],
            verbose=True,
            
        )
    
    @task
    def  research_task(self) -> Task:
        return  Task(config=self.tasks_config["research_task"])
    @task
    def fundamental_task(self) -> Task:
        return  Task(config=self.tasks_config["fundamental_task"])
    @task
    def sentiment_task(self) -> Task:
        return  Task(config=self.tasks_config["sentiment_task"])
    @task
    def risk_task(self) -> Task:
        return  Task(config=self.tasks_config["risk_task"],output_pydantic=RiskAssessment)
    @task
    def report_task(self) -> Task:
        return  Task(config=self.tasks_config["report_task"])

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
   
    @crew
    def crew(self) -> Crew:
        """Creates the StockAnalyst crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            tracing=True,
            
        )
