from typing import Annotated, Sequence, TypedDict, Literal
from langgraph.types import Command
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage

from dotenv import load_dotenv
import os

from langchain_community.tools import TavilySearchResults
import json

# Load environment variables from .env file
# load_dotenv()
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
load_dotenv()
for _key in ("GOOGLE_API_KEY", "TAVILY_API_KEY"):
    if not os.getenv(_key):
        raise RuntimeError(f"{_key} is missing. Add it to .env locally or to Secrets on Streamlit Cloud.")

def check_weather(location: str) -> str:
    '''Return the weather forecast for the specified location.'''
    return f"It's always sunny in {location}"

def json_validator(string: str):
    """Validates if the given string is a correctly formatted JSON."""
    try:
        json.loads(string)
        return "Valid JSON"
    except json.JSONDecodeError as e:
        return "Invalid JSON: " +  str(e)

tavily_search = TavilySearch(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    # include_domains=[...],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)
tools = [check_weather, tavily_search]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

research_agent = create_react_agent(
    tools=tools, 
    model=llm,
    prompt=
    """
        You are a research-driven agent specializing in gathering detailed and accurate information on a given topic. Your primary objective is to compile comprehensive datasets that will inform a chart generator agent in creating effective visual representations. Since the chart generator cannot gather any data itself, it is your responsibility to provide all the necessary information, including the chart type and the exact numbers to be used in each chart. Recommend the most suitable chart types (e.g., bar chart, line chart, pie chart) based on the nature of the data to maximize clarity and insight. If sufficient data for a specific chart is unavailable, omit that chart entirely.

        Follow these guidelines:

            Make only one tool call at a time and wait for the response before proceeding.
            Be prepared to refine your search queries or conduct additional research based on feedback or new requirements from the chart generator.

        Ensure a systematic approach to gathering and presenting data, always prioritizing clarity and precision.
    """
)

chart_agent = create_react_agent(
    model=llm,
    tools=[json_validator],
    prompt=
    """
        You are a chart_generator agent. Your task is to create a JSON array representing charts based on data and specifications provided by the researcher agent. 

        1. **Analyze Input**: 
        - If data is incomplete, request missing details, specifying exactly what is needed. It is important to include the text ~QUESTION_TO_RESEARCHER~ in your response when asking for missing information.

        - Avoid asking for already provided information.

        2. **Generate JSON**:
        - Create an array where each object represents a chart. Here is a sample output:
            ```json
            {
                chartsArray: [
                    {
                        type: "bar",
                        title: "Top EDM Songs by Streaming Numbers",
                        xAxisLabel: "Streaming Numbers (in billions)",
                        yAxisLabel: "Songs",
                        descriptionOfLabel: "song",
                        chartData: [
                            { label: "Levels", value: 380000000 },
                            { label: "Titanium", value: 1500000000 },
                            { label: "Wake Me Up", value: 1500000000 },
                            { label: "Animals", value: 1000000000 },
                            { label: "This Is What It Feels Like", value: 300000000 }
                        ],
                        chartFootNote: "<Go through the research agent's output and include all research findings that pertain to this chart here>"
                    },
                    {
                        type: "pie",
                        title: "Distribution of Top EDM Songs by Artist",
                        descriptionOfLabel: "artist",
                        chartData: [
                            { label: "Avicii", value: 40 },
                            { label: "David Guetta", value: 30 },
                            { label: "Martin Garrix", value: 30}
                        ],
                        chartFootNote: "<Go through the research agent's output and include all research findings that pertain to this chart here>"
                    }
                ],
                otherResearchFindings: "<Go through the research agent's output and include any research findings that were not already included in the chart foot notes>"
            }
            ```

        3. **Validate Output**:
        - Use the `json_validator` tool to ensure the JSON is valid.
        - Regenerate JSON until valid.

        **Output only the JSON in the specified format, without any additional text.**
    """
)


# class GraphState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]

# graph_builder = StateGraph(GraphState)
# graph_builder.add_node("researcher", research_agent)
# graph_builder.add_node("chartBuilder", chart_agent)

# graph_builder.add_edge(START, "researcher")
# graph_builder.add_edge("researcher", "chartBuilder")

# def chart_to_research_condition(state: GraphState) -> str:
#     chart_content = state["messages"][-1].content
#     if "QUESTION_TO_RESEARCHER" in chart_content:
#         return "research_more"
#     else:
#         return "path_end"
    
# graph_builder.add_conditional_edges(
#     "chartBuilder", 
#     chart_to_research_condition, 
#     {"research_more": "researcher", "path_end": END}
# )

# graph = graph_builder.compile()
# graph = chart_agent
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _text(content) -> str:
    """Gemini can return content as a list of blocks; flatten to a string."""
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return content or ""


def researcher_node(state: GraphState):
    messages = list(state["messages"])
    if not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(content=(
            "The chart generator needs more information (see its last message). "
            "Research it and reply with the exact data."
        )))
    result = research_agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


def chart_node(state: GraphState):
    messages = list(state["messages"])
    if not isinstance(messages[-1], HumanMessage):
        messages.append(HumanMessage(content=(
            "Using the research above, generate the chart JSON in the required format."
        )))
    result = chart_agent.invoke({"messages": messages})
    return {"messages": result["messages"]}


graph_builder = StateGraph(GraphState)
graph_builder.add_node("researcher", researcher_node)
graph_builder.add_node("chartBuilder", chart_node)

graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", "chartBuilder")


def chart_to_research_condition(state: GraphState) -> str:
    chart_content = _text(state["messages"][-1].content)
    if "QUESTION_TO_RESEARCHER" in chart_content:
        return "research_more"
    return "path_end"


graph_builder.add_conditional_edges(
    "chartBuilder",
    chart_to_research_condition,
    {"research_more": "researcher", "path_end": END},
)

graph = graph_builder.compile()