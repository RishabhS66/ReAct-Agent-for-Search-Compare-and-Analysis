import gradio as gr

# Import necessary libraries
from langchain import LLMChain, PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SerpAPIWrapper
from langchain.llms.base import LLM
from langchain_google_vertexai import ChatVertexAI
import os

os.environ['SERPAPI_API_KEY'] = os.getenv('SERPAPI_API_KEY')

if not os.environ['SERPAPI_API_KEY']:
    print("SERPAPI_API_KEY is not set. Please configure the environment variable.")

llm = ChatVertexAI(
    model="gemini-1.5-flash-002",
    temperature=0
)

# Load the search tool using SerpAPI
search_tool = SerpAPIWrapper()

search_instance = Tool(
    name="Search",
    description="A tool to search for information on the internet.",
    func=search_tool.run
)

# Custom tool for general comparison
comparison_prompt_template = """
Compare the following items in terms of the given category:
Items: {items}
Category: {category}
Provide a detailed comparison.

Comparison:
"""
comparison_prompt = PromptTemplate(template=comparison_prompt_template, input_variables=["items", "category"])
comparison_chain = LLMChain(llm=llm, prompt=comparison_prompt)

def compare_items(query: str) -> str:
    """Compare items based on the query."""
    all_input = query.split(":")

    # Create a prompt using the extracted items and category
    items_str = all_input[0]
    category = all_input[1]
    comparison_input = {"items": items_str, "category": category}

    # Run the comparison chain with the prompt input
    comparison_result = comparison_chain.run(comparison_input)

    # Return the formatted comparison result
    return f"{comparison_result}\n-----\nComparison completed. Proceed to analysis of the comparison."

compare_instance = Tool(
    name="Compare",
    description="A tool to compare multiple items based on a specific category.",
    func=compare_items
)

analysis_prompt_template = """
Analyze the following content to extract the key information and insights:
Content: {content}
Provide a concise summary.

Summary:
"""

analysis_prompt = PromptTemplate(template=analysis_prompt_template, input_variables=["content"])
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

def analyze_results(results: str) -> str:
    """Analyze search and comparison results"""
    # If query is empty or doesn't request comparison, provide a sample query
    if not results:
        return "No data provided for analysis."

    # Run analysis
    analysis_input = {"content": results}
    analysis_summary = analysis_chain.run(analysis_input)

    return f"Analysis Summary:\n{analysis_summary}"

analyze_instance = Tool(
    name="Analyze",
    description="A tool to analyze and summarize content before presenting it to user.",
    func=analyze_results
)


agent_tools = [search_instance, compare_instance, analyze_instance]
react_agent = initialize_agent(agent_tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

def process_query(query: str, max_steps: int = 100) -> str:
    response = react_agent({"input": query, "max_steps": max_steps})
    return response

# Gradio UI
def gradio_interface(query):
    result = process_query(query)
    return result['output']

# Create Gradio Interface
gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="Interactive ReAct Agent Interface",
    description="Input a query for the ReAct agent to process. The final answer will be displayed."
).launch(share=True)
