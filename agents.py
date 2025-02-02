import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from crewai_tools import SerperDevTool
from tools import get_stock_prices

load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
search_tool = SerperDevTool()


Senior_Financial_Strategist = Agent(
    role="Senior Financial Strategist",
    goal="Review the data generated from  both the tools that include data and recent news about teh company and compile a final report",
    backstory="You are a senior financial strategist with expertise in risk assessment, market trends, and high-value investments.",
    verbose=True,
    allow_delegation=False,
    tools=[get_stock_prices, search_tool]  # Both tools are assigned to the manager agent
)
