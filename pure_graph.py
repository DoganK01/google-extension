from typing import Annotated, Optional, Literal, Dict, Any
from typing_extensions import TypedDict

from langchain_openai import AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

import asyncio
import os
import requests
import psutil
import logging
import json

from constants import SYSTEM_PROMPT
from settings import settings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

class State(TypedDict):
    """Defines the state structure for the graph execution."""
    messages: Annotated[list, add_messages]  # Stores conversation history
    url: Optional[str]  # The extracted URL
    domain: Optional[str]  # The core domain extracted from the URL
    connection_checking: bool  # Whether a Chrome connection is active
    iteration: int  # Count of interactions

# Initialize memory and graph
memory = MemorySaver()
graph_builder = StateGraph(State)

# Define tools
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# Initialize LLM with tools
llm = AzureChatOpenAI(
    azure_deployment=settings.AZURE_DEPLOYEMENT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
)
llm_with_tools = llm.bind_tools(tools)

async def tools_condition(
    state: State,
    messages_key: str = "messages",
) -> Literal["tools", "__end__"]:
    """Use in the conditional_edge to route to the ToolNode if the last message

    has tool calls. Otherwise, route to the end.

    Args:
        state (State): The state to check for
            tool calls. Must have a list of messages (MessageGraph) or have the
            "messages" key (StateGraph).

    Returns:
        The next node to route to.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif isinstance(state, dict) and (messages := state.get(messages_key, [])):
        ai_message = messages[-1]
    elif messages := getattr(state, messages_key, []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0 and state["iteration"] < 4:
        return "tools"
    return "__end__"

async def connection(state: State) -> Dict[str, bool]:
    """Checks if Chrome is running to determine connection status.
    
    Args:
        state (State): The current state of execution.
    
    Returns:
        Dict[str, bool]: The updated state with connection status.
    """
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'chrome.exe':
            return {"connection_checking": True}
    return {"connection_checking": False}

async def chatbot(state: State) -> Dict[str, Any]:
    """Processes user input using the chatbot and updates the message history.
    
    Args:
        state (State): The current state of execution.
    
    Returns:
        Dict[str, Any]: The updated state with new messages.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "URL: {domain} \n\nContext: {messages}")
    ])
    
    chain = {
        "messages": RunnablePassthrough(),
        "domain": lambda x: state["domain"]
    } | prompt | llm_with_tools
    message = await chain.ainvoke(state["messages"])
    return {"messages": [message], "iteration": state["iteration"] + 1}

async def connection_checker(state: State) -> Literal["continue", "back"]:
    """Determines the next step based on connection status.
    
    Args:
        state (State): The current state of execution.
    
    Returns:
        Literal["continue", "back"]: The next node to route to.
    """
    return "continue" if state["connection_checking"] else "back"

async def get_current_url(state: State) -> Dict[str, Optional[str]]:
    """Fetches the current URL from Chrome's DevTools Protocol.
    
    Args:
        state (State): The current state of execution.
    
    Returns:
        Dict[str, Optional[str]]: The updated state with the extracted URL.
    """
    try:
        response = requests.get("http://localhost:9222/json")
        tabs = response.json()
        return {"url": tabs[0]['url']} if tabs else {"url": None}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error accessing Chrome Debugger: {e}")
        return {"url": None}

async def extract_core_domain(state: State) -> Dict[str, Optional[str]]:
    """Extracts the core domain from a full URL.
    
    Args:
        state (State): The current state of execution.
    
    Returns:
        Dict[str, Optional[str]]: The updated state with the extracted domain.
    """
    if state["url"]:
        try:
            return {"domain": state["url"].split('/')[2]}
        except IndexError:
            logging.error(f"Failed to extract domain from URL: {state["url"]}")
            return {"domain": None}
    return {"domain": None}

# Define graph structure
graph_builder.add_node("connection", connection)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("get_current_url", get_current_url)
graph_builder.add_node("extract_core_domain", extract_core_domain)
graph_builder.add_node("tools", ToolNode(tools=[search_tool]))

graph_builder.add_conditional_edges("connection", connection_checker, {"continue": "get_current_url", "back": "connection"})
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("get_current_url", "extract_core_domain")
graph_builder.add_edge("extract_core_domain", "chatbot")
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("connection")
graph = graph_builder.compile(checkpointer=memory)

#[{"role": "user", "content": user_input}]

async def stream_graph_updates(user_input: str):
    config = {"configurable": {"thread_id": "1"}}
    async for event in graph.astream({"messages": user_input, "iteration": 0}, config):
            if 'chatbot' in event and 'messages' in event['chatbot'] and event['chatbot']['messages']:
                last_message_content  = event['chatbot']['messages'][-1].content
                print(json.dumps({"Last Message Content": last_message_content}, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            asyncio.run(stream_graph_updates(user_input))
        except:
            # fallback if input() is not available
            user_input = "Etsy"
            print("User: " + user_input)
            asyncio.run(stream_graph_updates(user_input))
            break
