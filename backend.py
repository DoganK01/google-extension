from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator

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
import re
import uvicorn

from constants import SYSTEM_PROMPT
from settings import settings

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("gorgeous_chatbot_backend")

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

app = FastAPI(title="Gorgeous Chatbot Generator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify only your extension's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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


class RequestData(BaseModel):
    message: str
    url: HttpUrl

    @field_validator('message', mode="before")
    def non_empty_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty")
        return value
    @field_validator('url', mode="before")
    def convert_url_to_string(cls, value: HttpUrl) -> str:
        return str(value)  # Convert HttpUrl to str

class ResponseData(BaseModel):
    answer: str

async def extract_core_domain(state: State) -> str:
    try:
        pattern = re.compile(r"^(https?://[^/]+)")
        #url = str(state["url"])
        match = pattern.match(state["url"])
        if match:
            return {"domain": match.group(1)}
        raise ValueError("Invalid URL provided")
    except Exception as e:
        logger.exception("Processing error")
        raise HTTPException(status_code=500, detail="Generation error")

# Define graph structure
graph_builder.add_node("connection", connection)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("extract_core_domain", extract_core_domain)
graph_builder.add_node("tools", ToolNode(tools=[search_tool]))

graph_builder.add_conditional_edges("connection", connection_checker, {"continue": "extract_core_domain", "back": "connection"})
graph_builder.add_conditional_edges("chatbot", tools_condition)

graph_builder.add_edge("extract_core_domain", "chatbot")
graph_builder.add_edge("tools", "chatbot")

graph_builder.set_entry_point("connection")

graph = graph_builder.compile(checkpointer=memory)


async def stream_graph_updates(input):
    config = {"configurable": {"thread_id": "1"}}
    async for event in graph.astream({"messages": input.message, "url": str(input.url), "iteration": 0}, {"recursion_limit": 250}, config):
            #print("\n\nPRINTINGGGGG EVENT : \n\n", event)
            if 'chatbot' in event and 'messages' in event['chatbot'] and event['chatbot']['messages']:
                last_message_content  = event['chatbot']['messages'][-1].content
                return last_message_content
                #print(json.dumps({"Last Message Content": last_message_content}, indent=4, ensure_ascii=False))

@app.post("/generate", response_model=ResponseData)
async def generate_response(data: RequestData) -> Any:
    try:
        logger.info(f"Received message: '{data.message}' from domain: {data.url}")
        generated_answer = await stream_graph_updates(data)
    except Exception as e:
        logger.exception("Processing error")
        raise HTTPException(status_code=500, detail="Generation error")

    return ResponseData(answer=generated_answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
