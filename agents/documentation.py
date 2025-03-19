import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, TypedDict, Any

import trafilatura
from trafilatura.settings import use_config

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from langchain_community.tools import DuckDuckGoSearchResults

# Load ENV variables
load_dotenv()

# Agent State Model
class AgentState(TypedDict):
    """State for the documentation research agent."""
    # Input information
    package_name: str  # The name of the package, library, SDK or API to research
    functionality: str  # The specific functionality to find documentation for

    # Agent communication
    current_step: str  # Current step the agent is working on
    messages: List[Dict]  # All messages in the conversation
    
    # Enhanced search capabilities
    search_terms: List[str]  # Specific terms to search for
    functionality_context: Dict  # Context about the functionality
    
    # Output
    code_examples: List[Dict]  # Working code examples
    structured_documentation: Dict  # Final structured documentation

    # Working memory
    search_results: List[Dict]  # Results from search operations
    documentation_urls: List[str]  # Relevant documentation URLs
    documentation_content: List[Dict]  # Extracted documentation content
    
    # Iteration and evaluation
    iteration_count: int  # Number of improvement iterations
    quality_score: Optional[int]  # Evaluation score of documentation quality
    evaluation_feedback: Optional[str]  # Feedback for improvement
    
    # Errors and debugging
    errors: List[str]  # Errors encountered during processing

# Structured Output Models
class FunctionalityAnalysis(BaseModel):
    """Model for analyzing functionality and generating search terms."""
    search_terms: List[str] = Field(description="Technical terms for API docs")
    alternative_names: List[str] = Field(description="Alternative names for the functionality")
    related_components: List[str] = Field(description="Related methods, classes, or components")

class DocumentationStructure(BaseModel):
    """Model for structured documentation output."""
    overview: str = Field(description="Brief overview of the functionality")
    references: List[str] = Field(description="References to official documentation")
    key_concepts: List[str] = Field(description="List of key concepts with explanations")
    usage_patterns: str = Field(description="Description of how to use the functionality")
    installation: str = Field(description="Installation and setup instructions if needed")
    code_examples: List[str] = Field(description="Complete formatted runnable code examples with explanations")
  
class DocumentationEvaluation(BaseModel):
    """Model for documentation quality evaluation."""
    feedback: str = Field(description="Specific feedback on how to improve")
    completeness_score: int = Field(description="Completeness score from 1-10")
    quality_score: int = Field(description="Quality score from 1-10, where 10 is excellent")
    sufficient: bool = Field(description="Whether the documentation is sufficient or needs improvement")

# LLM getter
def get_llm(model="gpt-4o-mini", temperature=0.1):
    """Get an instance of the LLM with the specified parameters."""
    return ChatOpenAI(model=model, temperature=temperature)

# Tools getter
def get_search_tools():
    """Get search tools for the agent."""
    return [DuckDuckGoSearchResults(num_results=3, output_format="list")]

async def fetch_webpage_content(url: str) -> Dict:
    """
    Fetch content from a webpage and return it as text.
    
    Args:
        url: The URL of the webpage to fetch
        
    Returns:
        Dict with the extracted text content from the webpage
    """
    try:
        downloaded = trafilatura.fetch_url(url)

        if downloaded:
            extracted_text = trafilatura.extract(downloaded, output_format="markdown", with_metadata=False, include_formatting=False, include_images=False, include_links=False)

            if extracted_text:
                return {
                    "url": url,
                    "text": extracted_text,
                }
            else:
                return {"error": "Failed to extract text content using trafilatura."}
        else:
            return {"error": "Failed to download the webpage using trafilatura."}

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Node functions
async def analyze_functionality(state: AgentState) -> AgentState:
    """
    Analyze the requested functionality to generate targeted search terms and provide context.
    This step helps in generating more precise and relevant search queries.
    """
    state["current_step"] = "analyzing_functionality"
    
    system_message = SystemMessage(content="""
    You are a Documentation Analysis Expert specialized in understanding technical concepts and libraries.
    Your task is to analyze a requested functionality within a specific package/library and generate:
    
    1. A list of specific search terms that would help find relevant documentation
    2. Alternative names or related concepts for this functionality
    3. Related methods, classes, or components that might be involved
    
    Provide this information in a structured format that will be used to perform targeted documentation searches.
    """)
    user_message = HumanMessage(content=f"""
    Please analyze the following request for documentation:
    
    Package/Library/SDK/API: {state["package_name"]}
    Specific Functionality: {state["functionality"]}
    
    Provide a detailed analysis.
    """)

    messages = [system_message, user_message]
    
    try:
        # LLM call
        llm = get_llm()
        analyzer = llm.with_structured_output(FunctionalityAnalysis)
        analysis_result = await analyzer.ainvoke(messages)

        # Store the consolidated search terms
        state["search_terms"] = analysis_result.search_terms
        # Store the analysis results
        state["functionality_context"] = analysis_result.model_dump()
    except Exception as e:
        state["errors"] = state.get("errors", []) + [f"Error analyzing functionality: {str(e)}"]
        # Create fallback search terms if analysis fails
        state["search_terms"] = [
            f"{state['package_name']} {state['functionality']} documentation",
            f"{state['package_name']} {state['functionality']} tutorial",
            f"{state['package_name']} {state['functionality']} guide",
            f"{state['package_name']} {state['functionality']} examples"
        ]
    
    return state

async def search_for_documentation(state: AgentState) -> AgentState:
    """
    Search for documentation related to the package and functionality
    using the enhanced search terms from the functionality analysis.
    """
    state["current_step"] = "searching_for_documentation"
    
    # Search queries
    queries = []
    
    # Start with combination of package name and functionality
    base_query = f"{state['package_name']} {state['functionality']} documentation"
    queries.append(base_query)
    
    # Adding more specific queries using the analyzed search terms
    for term in state["search_terms"][:4]:  # Use top 4 search terms
        queries.append(f"{state['package_name']} {term} documentation")
    
    # Deduplicate queries
    queries = list(set(queries))[:3]  # Limit to top 3 most relevant queries
    
    search_results = []
    search_tools = get_search_tools()
    search_tool = search_tools[0]
    
    # Execute searches
    for query in queries:
        results = await search_tool.ainvoke(query)
        search_results.extend(results)
    
    # Store search results
    state["search_results"] = search_results
    
    # Extract potentially relevant documentation URLs
    doc_urls = []
    for result in search_results:
        if isinstance(result, dict):
            if "link" in result:
                doc_urls.append(result["link"])
            elif "url" in result:
                doc_urls.append(result["url"])
    
    # Deduplicate URLs
    state["documentation_urls"] = list(set(doc_urls))[:5]  # Limit to top 5 URLs

    return state

async def extract_documentation_content(state: AgentState) -> AgentState:
    """
    Extract and process content from documentation URLs.
    """
    state["current_step"] = "extracting_documentation"
    
    documentation_content = []
    
    # Processing documentation URLs
    for url in state["documentation_urls"]:
        try:
            content_result = await fetch_webpage_content(url)
                
            documentation_content.append({
                "url": url,
                "content": content_result.get("text", ""),
            })
        except Exception as e:
            state["errors"] = state.get("errors", []) + [f"Error processing {url}: {str(e)}"]
    
    state["documentation_content"] = documentation_content    

    return state

async def analyze_and_structure_documentation(state: AgentState) -> AgentState:
    """
    Analyze the documentation content and structure it in a useful format.
    """
    state["current_step"] = "analyzing_documentation"
    
    # Preparing context for the LLM
    context = "Documentation Sources:\n"    
    for doc in state["documentation_content"]:
        context += f"\nSource: {doc['url']}\n Content: {doc['content']}\n"
    
    system_message = SystemMessage(content="""
    You are a Documentation Research Agent specialized in analyzing and structuring technical documentation.
    Your task is to analyze the provided documentation and create a well-structured, easy-to-understand guide
    about the specific functionality requested.
    
    The documentation should include:
    1. Overview of the functionality
    2. Installation and setup instructions (if relevant)
    3. Key concepts and components
    4. Usage patterns and API details
    5. Working code examples (at least 2-3 examples)
    6. Common issues and solutions
    7. References to official documentation
    
    Focus only on the specific functionality requested, not the entire package/library.
    Ensure the code examples are complete, runnable, and well-commented.
    """)
    user_message = HumanMessage(content=f"""
    Please analyze the following documentation content and create a structured guide for using
    {state["functionality"]} in {state["package_name"]}.
    
    {context}
    """)
    
    messages = [system_message, user_message]
    
    # LLM call
    llm = get_llm()
    analyzer = llm.with_structured_output(DocumentationStructure)
    structured_doc = await analyzer.ainvoke(messages)
    
    # Store structured documentation and code examples
    state["structured_documentation"] = structured_doc.model_dump()
        
    return state

async def evaluate_documentation(state: AgentState) -> AgentState:
    """
    Evaluate the quality of the documentation and provide feedback for improvement.
    """
    state["current_step"] = "evaluating_documentation"
    # Initialize iteration count if not present
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    # Skip evaluation if we've reached the maximum iterations
    if state["iteration_count"] >= 3:
        state["quality_score"] = 10  # Assume we're done after 3 iterations
        state["evaluation_feedback"] = "Maximum iterations reached."
        return state
    
    # Get the structured documentation
    doc = state["structured_documentation"]
    
    system_message = SystemMessage(content="""
    You are a Documentation Quality Evaluator specialized in technical documentation.
    Your task is to evaluate the provided documentation and provide a quality score and feedback.
    
    Evaluate the documentation on:
    1. Completeness - Does it cover all key aspects of the functionality?
    2. Clarity - Is the information presented clearly and logically?
    3. Examples - Are the code examples useful, complete, and well-explained?
    4. Technical accuracy - Does the documentation appear technically accurate?
    5. Usability - Would a developer be able to use the functionality after reading this?
    
    Provide a quality score from 1-10 and specific feedback on how to improve the documentation.
    """)    
    user_message = HumanMessage(content=f"""
    Please evaluate this documentation for {state["functionality"]} in {state["package_name"]}:
    
    Overview: {doc.get("overview", "Missing")}
    
    Installation: {doc.get("installation", "Missing")}
    
    Key Concepts: {doc.get("key_concepts", "Missing")}
    
    Usage Patterns: {doc.get("usage_patterns", "Missing")}
    
    Code Examples: {len(doc.get("code_examples", []))} examples provided
    
    References: {len(doc.get("references", []))} references provided
    """)
    
    try:
        # LLM call
        llm = get_llm()
        evaluator = llm.with_structured_output(DocumentationEvaluation)
        evaluation = await evaluator.ainvoke([system_message, user_message])
        
        state["quality_score"] = evaluation.quality_score
        state["evaluation_feedback"] = evaluation.feedback
    except Exception as e:
        state["errors"] = state.get("errors", []) + [f"Error evaluating documentation: {str(e)}"]
        # Provide fallback evaluation
        state["quality_score"] = 5
        state["evaluation_feedback"] = "Unable to evaluate properly due to an error."
    
    return state

# Workflow 
def build_documentation_research_graph():
    """
    Build the workflow graph for the documentation research agent.
    """
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_functionality", analyze_functionality)
    workflow.add_node("search_for_documentation", search_for_documentation)
    workflow.add_node("extract_documentation_content", extract_documentation_content)
    workflow.add_node("analyze_and_structure_documentation", analyze_and_structure_documentation)
    
    # Add edges
    workflow.add_edge(START, "analyze_functionality")
    workflow.add_edge("analyze_functionality", "search_for_documentation")
    workflow.add_edge("search_for_documentation", "extract_documentation_content")
    workflow.add_edge("extract_documentation_content", "analyze_and_structure_documentation")
    workflow.add_edge("analyze_and_structure_documentation", END)
    
    # Compile the graph
    return workflow.compile()

# Main function to run the documentation research agent
async def run_documentation_research_agent(package_name: str, functionality: str) -> Dict:
    """
    Run the documentation research agent for a specific package and functionality.
    
    Args:
        package_name: The name of the package, library, SDK or API to research
        functionality: The specific functionality to find documentation for
        
    Returns:
        A dictionary containing structured documentation and code examples
    """
    # Initialize the state
    initial_state = {
        "errors": [],
        "messages": [],
        "code_examples": [],
        "search_terms": [],
        "current_step": "",
        "search_results": [],
        "quality_score": None,
        "documentation_urls": [],
        "functionality_context": {},
        "documentation_content": [],
        "evaluation_feedback": None,
        "package_name": package_name,
        "functionality": functionality,
        "structured_documentation": {},
    }
    
    # Build and run
    graph = build_documentation_research_graph()
    result = await graph.ainvoke(initial_state)
    
    return result["structured_documentation"]