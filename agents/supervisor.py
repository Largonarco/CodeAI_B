import asyncio
from dotenv import load_dotenv
from pymongo import AsyncMongoClient
from typing_extensions import TypedDict
from typing import Literal, Dict, Optional

from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import AsyncMongoDBSaver
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from .pr_review import analyze_pull_request
from .documentation import run_documentation_research_agent

load_dotenv()

# Models
class MultiAgentState(MessagesState):
    """State for the multi-agent supervisor workflow"""
    next: str  # Next agent to route to
    results: Dict  # Results from agent operations
    query: Optional[str]  # For general agent
    repo_url: Optional[str]  # For PR review agent
    pr_number: Optional[int]  # For PR review agent
    github_token: Optional[str]  # For PR review agent
    package_name: Optional[str]  # For documentation agent
    functionality: Optional[str]  # For documentation agent

class RouterResponse(TypedDict):
    """Response structure from supervisor"""
    next: Literal["documentation_agent", "pr_review_agent", "general_agent", "FINISH"]
    params: Optional[Dict]  # Either DocumentationParams, PRReviewParams, or GeneralParams

class DocumentationParams(TypedDict):
    package_name: str
    functionality: str

class PRReviewParams(TypedDict):
    repo_url: str
    pr_number: int
    github_token: Optional[str]

class GeneralParams(TypedDict):
    query: str


# Supervisor setup
members = ["documentation_agent", "pr_review_agent", "general_agent"]
options = members + ["FINISH"]

system_prompt = f"""
You are a supervisor managing a team of specialized agents: {members}.
Based on the user request, determine:
1. Which agent should handle the task (or FINISH if complete)
2. The specific parameters required for that agent

Available agents and their required parameters:
- documentation_agent:
  - package_name (str): Name of the package/library
  - functionality (str): Specific functionality to document
- pr_review_agent:
  - repo_url (str): GitHub repository URL
  - pr_number (int): Pull request number
  - github_token (str, optional): GitHub authentication token
- general_agent:
  - query (str): The user's question or request

Analyze the user's message and the current state:
- If this is the initial request, route to the appropriate agent with parameters.
- If responding to a user query an agent has completed it's task and explicitly mentioned it (check recent messages), route to FINISH.
- If parameters cannot be determined, include an error message in the response.
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# Supervisor node
async def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    if state["query"]:
        state["messages"] = state["messages"] + [HumanMessage(content=state["query"])]
    supervisor_messages = [
        SystemMessage(content=system_prompt.format(members=members)),
    ] + state["messages"]
    
    try:
        response = await llm.with_structured_output(RouterResponse).ainvoke(supervisor_messages)

        goto = response["next"]
        state["next"] = goto
        
        if response["params"]:
            if goto == "general_agent":
                params = GeneralParams(**response["params"])

                state["query"] =  params["query"]
            elif goto == "documentation_agent":
                params = DocumentationParams(**response["params"])

                state["package_name"] = params["package_name"]
                state["functionality"] = params["functionality"]
            elif goto == "pr_review_agent":
                params = PRReviewParams(**response["params"])

                state["repo_url"] = params["repo_url"]
                state["pr_number"] = params["pr_number"]
                state["github_token"] = params.get("github_token")        
    except Exception as e:
        state["messages"] = state["messages"] + [SystemMessage(content=f"Supervisor error: {str(e)}")]
        
    return state
    
def should_continue(state: MultiAgentState) -> Literal["summarize", "documentation_agent", "pr_review_agent", "general_agent", "FINISH"]:
    """Return the next node to execute."""
    # If there are more than six messages, then we summarize the conversation
    if len(state["messages"]) > 10:
        return "summarize"
    elif state["next"] == "documentation_agent":
        return "documentation"
    elif state["next"] == "pr_review_agent":
        return "pr_review"
    elif state["next"] == "general_agent":
        return 'general'
 
    # Otherwise we can just end
    return "FINISH"
    
async def summarize_conversation_node(state: MultiAgentState) -> MultiAgentState:
    messages = state["messages"]
    latest_message = messages.pop()
    summariser_messages = []
    summary_message = "Create a summary of the entire conversation above with a higher emphasis towards retaining factual information pertaining to the user, for example: the user's name, likings, etc."

    if isinstance(latest_message, HumanMessage):
        summariser_messages = messages 
    else:
        summariser_messages = messages + [latest_message] 
        
    response = await llm.ainvoke(summariser_messages + [HumanMessage(content=summary_message)])
    
    # Remove all existing messages and replace with summary
    delete_messages = [RemoveMessage(id=m.id) for m in summariser_messages]

    if isinstance(latest_message, HumanMessage):
        state["messages"] = delete_messages + [SystemMessage(content=f"Conversation Summary: {response.content}")] + [latest_message]
    else:
        state["messages"] = delete_messages + [SystemMessage(content=f"Conversation Summary: {response.content}")]

    state["query"] = None
    return state
    
# General Agent
async def general_agent_node(state: MultiAgentState) -> MultiAgentState:
    if not state.get("query"):
        state["messages"] = state["messages"] + [SystemMessage(content="Error: Missing query parameter for general agent")]
    
    try:
        messages = [SystemMessage(content="You are a helpful AI assistant. Provide a clear and concise response to the user's query.")] + state["messages"] + [HumanMessage(content=state["query"])]
        result = await llm.ainvoke(messages)
        
        state["messages"] = state["messages"] + [SystemMessage(content=str(result.content))]
        state["results"] = {**state.get("results", {}), "general_response": str(result.content)}
    except Exception as e:
        state["messages"] = state["messages"] + [SystemMessage(content=f"Error in general agent: {str(e)}")]
        
    state["query"] = None
    return state

# Documentation Agent 
async def documentation_agent_node(state: MultiAgentState) -> MultiAgentState:
    if not state.get("package_name") or not state.get("functionality"):
        state["messages"] = state["messages"] + [SystemMessage(content="Error: Missing package_name or functionality parameters")] 
    
    try:
        result = await run_documentation_research_agent(
            package_name=state["package_name"],
            functionality=state["functionality"]
        )
        
        state["messages"] = state["messages"] + [SystemMessage(content=f"Documentation research completed for {state['package_name']} - {state['functionality']}\n\nResult:\n{result}")]
        state["results"] = {**state.get("results", {}), "documentation": result}
    except Exception as e:
        state["messages"] = state["messages"] + [SystemMessage(content=f"Error in Documentation agent: {str(e)}")]

    state["query"] = None
    return state

# PR Review Agent 
async def pr_review_agent_node(state: MultiAgentState) -> MultiAgentState:
    if not state.get("repo_url") or not state.get("pr_number"):
        state["messages"] = state["messages"]  + [SystemMessage(content="Error: Missing repo_url or pr_number parameters")]
    
    try:
        result = await analyze_pull_request(
            repo_url=state["repo_url"],
            pr_number=state["pr_number"],
            github_token=state.get("github_token")
        )
        
        state["messages"] = state["messages"] + [SystemMessage(content=f"PR review completed for {state['repo_url']} PR#{state['pr_number']}\n\nResult:{result}")]
        state["results"] = {**state.get("results", {}), "pr_review": result}
    except Exception as e:
        state["messages"] = state["messages"] + [SystemMessage(content=f"Error in PR Review agent: {str(e)}")]
       
    state["query"] = None
    return state

# Workflow
def build_multi_agent_workflow(async_mongo_client: any) -> StateGraph:
    workflow = StateGraph(MultiAgentState)
    mongo_checkpointer = AsyncMongoDBSaver(async_mongo_client)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("general_agent", general_agent_node)
    workflow.add_node("pr_review_agent", pr_review_agent_node)
    workflow.add_node("summarize", summarize_conversation_node)
    workflow.add_node("documentation_agent", documentation_agent_node)
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "summarize": "summarize",
            "documentation": "documentation_agent",
            "pr_review": "pr_review_agent",
            'general': "general_agent",
            "FINISH": END
        }
    )
    workflow.add_edge("general_agent", END)
    workflow.add_edge("summarize", "supervisor")
    workflow.add_edge("pr_review_agent", END)
    workflow.add_edge("documentation_agent", END)
    
    return workflow.compile(checkpointer=mongo_checkpointer)

# Execute function
async def run_multi_agent_workflow(request: str, async_mongo_client: AsyncMongoClient, config: Dict[str, Dict[str, str]], github_token: Optional[str] = None) -> Dict:
    initial_state = {
        "results": {},
        "query": request,
        "repo_url": None,
        "pr_number": None,
        "next": "supervisor",
        "package_name": None,
        "functionality": None,
        "general_messages": [],
        "github_token": github_token,
    }
    
    try:
        workflow = build_multi_agent_workflow(async_mongo_client=async_mongo_client)
        result = await workflow.ainvoke(initial_state, config=config)

         # Converting LangChain messages to dictionaries
        serialized_messages = [
            {
                "role": msg.type, 
                "content": msg.content,
            }
            for msg in result["messages"]
        ]

        return {
            "results": result["results"],
            "messages": serialized_messages
        }
    except Exception as e:
        return {"error": str(e)}

