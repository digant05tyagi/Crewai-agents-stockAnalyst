import os
from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy.sql.operators import truediv

from agents import Senior_Financial_Strategist

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash',
                            verbose=True,
                            temperature=0.5,
                            google_api_key=os.getenv('GEMINI_API_KEY'))


stock_ticker = input("Enter the stock ticker symbol (e.g., AAPL, TSLA, AMZN): ").strip().upper()


final_report_task = Task(
    description=(
        "Follow these steps to complete the task:\n\n"
        "1. Retrieve the latest 5 news articles about {stock_ticker} using the `search_tool`.\n"
        "2. Gather all relevant data for {stock_ticker} using the `get_stock_prices` tool and create a table for the same.\n"
        "3. Analyze the data and news from steps 1 and 2 to generate a comprehensive report"
        "which should include:\n"
        "- **Data Indicators and Values**: Provide insights for each indicator.\n"
        "- **News Summaries**: Offer insights based on the news articles retrieved.\n"
        "- **Table for all the data collected"
    ),
    expected_output=(
        f"A detailed report summarizing the stock performance, "
        f"technical indicators, the impact of recent news, and a final recommendation "
        f"for {stock_ticker}. Additionally, provide a suggestion on whether it is "
        f"a good buy or sell opportunity."
    ),
    agent=Senior_Financial_Strategist,
)

# Define a Crew workflow
my_agentic_workflow = Crew(
    llm = llm,
    agents=[Senior_Financial_Strategist],  # Excluding manager agent from agents list
    tasks=[final_report_task],
    process=Process.sequential,  # Process agents in sequence to ensure correct input flow
    verbose=True,
)

# Run the workflow
my_agentic_workflow.kickoff()
