from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import TypedDict, Literal, List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END

# State
class ResearchState(TypedDict):
    topic: str  # Research topic
    messages: List[Any]  # Conversational messages
    iteration_count: int  # Iteraction count
    research_summary: str  # Research summary
    research_strategy: str  # Research strategy
    search_queries: List[str]  # Search queries
    search_results: List[Dict]  # Search results
    quality_score: Optional[int]  # Evaluation Quality Score
    evaluation_feedback: Optional[str]  # Evaluation Feedback

# Define tool schemas
class SearchQuery(BaseModel):
    query: str = Field(description="Search query to run against a web search engine")
    justification: str = Field(description="Why this query is relevant to the research topic")

class ResearchPlan(BaseModel):
    queries: List[SearchQuery] = Field(description="List of search queries to perform")
    
class ResearchStrategy(BaseModel):
    strategy: Literal["broad", "deep", "technical", "recent", "balanced"] = Field(
        description="The research strategy to employ"
    )
    justification: str = Field(
        description="Explanation for why this strategy is appropriate for the topic"
    )
    
class ResearchEvaluation(BaseModel):
    quality_score: int = Field(
        description="Quality score from 1-10, where 10 is excellent research",
    )
    coverage_score: int = Field(
        description="Coverage score from 1-10, where 10 is comprehensive coverage",
    )
    feedback: str = Field(
        description="Specific feedback on how to improve the research"
    )
    sufficient: bool = Field(
        description="Whether the research is sufficient or needs improvement"
    )

# LLM getter
def get_llm():
    """Initialize the LLM with OpenAI"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm
  

# Search tools getter
def get_search_tools():
    """Get search tools for the agent"""
    try:
        # Try to use Tavily if API key is available
        tavily_search = TavilySearchResults(max_results=5)
        return [tavily_search]
    except:
        # Fallback to DuckDuckGo which doesn't require API key
        ddg_search = DuckDuckGoSearchResults(num_results=5)
        return [ddg_search]

# Research agent 
def create_research_agent():
    """Create a LangGraph research agent that gathers information from the web"""
    llm = get_llm()
    search_tools = get_search_tools()
    
    # Create structured output for evaluation
    evaluator = llm.with_structured_output(ResearchEvaluation)
    # Create structured output for strategy selection
    strategy_selector = llm.with_structured_output(ResearchStrategy)
    
    # Nodes
    def determine_strategy(state: ResearchState):
        """Determine the best research strategy for the topic"""
        topic = state["topic"]
        
        # Generate strategy
        strategy_result = strategy_selector.invoke([
            SystemMessage(content="""
                You are a research strategy expert. Given a research topic, determine the best approach:
                - broad: For general topics that need wide coverage
                - deep: For specific topics that need in-depth analysis
                - recent: For topics where recent developments are crucial
                - balanced: For topics needing a mix of breadth and depth
                - technical: For scientific or technical topics that need specialized information
            """),
            HumanMessage(content=f"Determine the best research strategy for: {topic}")
        ])
        
        return {"research_strategy": strategy_result.strategy}
    
    def plan_research(state: ResearchState):
        """Create a research plan with multiple search queries"""
        topic = state["topic"]
        strategy = state["research_strategy"]
        feedback = state.get("evaluation_feedback", "")
        
        # Create structured output for planning
        planner = llm.with_structured_output(ResearchPlan)
        
        # Strategy-specific instructions
        strategy_instructions = {
            "balanced": "Create a mix of broad and deep queries for balanced coverage.",
            "recent": "Focus on recent developments, news, and cutting-edge information.",
            "broad": "Create diverse queries covering many different aspects of the topic.",
            "deep": "Create focused queries that dive into specific aspects of the topic in detail.",
            "technical": "Create queries focused on technical aspects, papers, and specialized information.",
        }
        
        instruction = strategy_instructions.get(strategy, strategy_instructions["balanced"])
        
        # Include feedback if available
        feedback_instruction = ""
        if feedback:
            feedback_instruction = f"\nPrevious feedback to address: {feedback}"
        
        # Generate plan
        plan_message = planner.invoke([
            SystemMessage(content=f"""
                You are a research planning assistant. Given a research topic, create 3-5 search queries 
                that would help gather comprehensive information on the topic.
                
                Research strategy: {strategy}
                {instruction}
                {feedback_instruction}
            """),
            HumanMessage(content=f"Create a research plan for the topic: {topic}")
        ])
        
        # Extract search queries
        search_queries = [query.query for query in plan_message.queries]
        
        return {"search_queries": search_queries}
    
    def perform_searches(state: ResearchState):
        """Execute searches for all queries"""
        queries = state["search_queries"]
        strategy = state["research_strategy"]
        all_results = []
        
        # Use the first search tool (either Tavily or DuckDuckGo)
        search_tool = search_tools[0]
        
        # Adjust search parameters based on strategy
        max_results = 5
        if strategy == "deep" or strategy == "technical":
            max_results = 8  # More results for deep research
        elif strategy == "broad":
            max_results = 3  # Fewer but more diverse results
            
        for query in queries:
            try:
                # Modify query based on strategy
                modified_query = query
                if strategy == "recent":
                    modified_query = f"{query} last year"
                elif strategy == "technical":
                    modified_query = f"{query} research paper technical"
                    
                results = search_tool.invoke(modified_query)
                # Limit results to max_results
                if isinstance(results, list) and len(results) > max_results:
                    results = results[:max_results]
                    
                all_results.append({"query": query, "modified_query": modified_query, "results": results})
            except Exception as e:
                all_results.append({"query": query, "results": f"Error in search: {str(e)}"})
        
        return {"search_results": all_results}
    
    def synthesize_research(state: ResearchState):
        """Synthesize search results into a comprehensive research summary"""
        topic = state["topic"]
        search_results = state["search_results"]
        strategy = state["research_strategy"]
        feedback = state.get("evaluation_feedback", "")
        
        # Create a prompt with all the search results
        result_text = ""
        for item in search_results:
            modified_query = item.get("modified_query", item["query"])
            result_text += f"\nQuery: {item['query']}"
            if modified_query != item["query"]:
                result_text += f" (Modified to: {modified_query})"
            result_text += f"\nResults: {str(item['results'])[:1000]}...\n"
        
        # Strategy-specific synthesis instructions
        strategy_instructions = {
            "broad": "Organize information to provide a wide overview of the topic with many different aspects covered.",
            "deep": "Focus on providing in-depth analysis of key aspects rather than broad coverage.",
            "technical": "Emphasize technical details, methodologies, and specialized information. Use more technical language.",
            "recent": "Highlight recent developments and current state of the field. Note how recent each source is.",
            "balanced": "Balance breadth and depth, covering key aspects thoroughly while maintaining broad coverage."
        }
        
        instruction = strategy_instructions.get(strategy, strategy_instructions["balanced"])
        
        # Include feedback if available
        feedback_instruction = ""
        if feedback:
            feedback_instruction = f"\nPrevious feedback to address: {feedback}"
        
        # Generate research summary
        summary_message = llm.invoke([
            SystemMessage(content=f"""
                You are a research synthesis expert. Analyze the provided search results and create
                a comprehensive, well-structured research summary on the topic. 
                
                Research strategy: {strategy}
                {instruction}
                {feedback_instruction}
                
                Include:
                - Key findings and main points
                - Different perspectives or approaches
                - Important facts, statistics, or examples
                - Areas of consensus and debate
                - Citations for important information (use the sources from search results)
                
                Format your response with clear headings, bullet points where appropriate, and
                a logical flow of information.
            """),
            HumanMessage(content=f"Create a research summary on '{topic}' based on these search results:\n{result_text}")
        ])
        
        return {"research_summary": summary_message.content}
    
    def evaluate_research(state: ResearchState):
        """Evaluate the quality of the research and provide feedback"""
        topic = state["topic"]
        summary = state["research_summary"]
        strategy = state["research_strategy"]
        iteration = state.get("iteration_count", 0)
        
        # Generate evaluation
        evaluation = evaluator.invoke([
            SystemMessage(content=f"""
                You are a research quality evaluator. Analyze the provided research summary and
                evaluate its quality based on the following criteria:
                
                - Comprehensiveness: Does it cover the important aspects of the topic?
                - Accuracy: Does the information appear accurate and well-sourced?
                - Clarity: Is the research clearly presented and well-organized?
                - Relevance: Is the content directly relevant to the topic?
                - Depth: Is there sufficient detail on important aspects?
                
                Research strategy was: {strategy}
                This is iteration #{iteration + 1} of the research process.
                
                Be honest but constructive in your feedback.
            """),
            HumanMessage(content=f"Evaluate this research summary on '{topic}':\n\n{summary}")
        ])
        
        return {
            "iteration_count": iteration + 1,
            "quality_score": evaluation.quality_score,
            "evaluation_feedback": evaluation.feedback,
        }
    
    def should_improve(state: ResearchState):
        """Determine if we should improve the research or finish"""
        score = state["quality_score"]
        iteration = state["iteration_count"]
        
        # If quality is high enough or we've done 3 iterations, we're done
        if score >= 8 or iteration >= 3:
            return "finish"
        else:
            return "improve"
    
    def generate_response(state: ResearchState):
        """Convert research summary into a message response"""
        summary = state["research_summary"]
        strategy = state["research_strategy"]
        score = state["quality_score"]
        iterations = state["iteration_count"]
        
        # Add metadata about the research process
        metadata = f"""
        ---
        Research Strategy: {strategy}
        Quality Score: {score}/10
        Research Iterations: {iterations}
        ---
        """
        
        full_response = f"{summary}\n\n{metadata}"
        return {"messages": [HumanMessage(content=full_response)]}
    
    # Workflow
    workflow = StateGraph(ResearchState)
    
    # Adding nodes
    workflow.add_node("determine_strategy", determine_strategy)
    workflow.add_node("plan_research", plan_research)
    workflow.add_node("perform_searches", perform_searches)  
    workflow.add_node("synthesize_research", synthesize_research)
    workflow.add_node("evaluate_research", evaluate_research)
    workflow.add_node("generate_response", generate_response)
    
    # Adding edges
    workflow.add_edge(START, "determine_strategy")
    workflow.add_edge("determine_strategy", "plan_research")
    workflow.add_edge("plan_research", "perform_searches")
    workflow.add_edge("perform_searches", "synthesize_research")
    workflow.add_edge("synthesize_research", "evaluate_research")
    # Adding conditional edge based on evaluation
    workflow.add_conditional_edges(
        "evaluate_research",
        should_improve,
        {
            "improve": "plan_research",  # Loop back to planning with feedback
            "finish": "generate_response"  # Proceed to final response
        }
    )
    workflow.add_edge("generate_response", END)
    
    # Compiling workflow
    return workflow.compile()

# Main function to run the research agent
def research_topic(topic):
    """Run the research agent on a given topic"""
    agent = create_research_agent()
    initial_state = {
        "topic": topic, 
        "messages": [], 
        "iteration_count": 0,
        "quality_score": None,
        "search_queries": [], 
        "search_results": [], 
        "research_summary": "",
        "research_strategy": "",
        "evaluation_feedback": "",

    }
    
    state = agent.invoke(initial_state)

    return {
        "summary": state["research_summary"],
        "strategy": state["research_strategy"],
        "quality_score": state["quality_score"],
        "iterations": state["iteration_count"]
    }

