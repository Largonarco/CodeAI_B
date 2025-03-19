import aiohttp
from datetime import datetime

from github import Github
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END

from typing import TypedDict, List, Dict, Any, Optional, Literal

# Load ENV variables 
load_dotenv()  

# Agent State Model
class CodeReviewState(TypedDict):
    """State for the code review workflow"""
    status: str
    repo_url: str
    pr_number: int
    files: List[Any]  # GitHub PR files
    started_at: str
    error: Optional[str]
    summary: Dict[str, int]
    analyzed_files: List[Dict]
    github_token: Optional[str]
    completed_at: Optional[str]
    file_contents: Dict[str, str]  # Full file contents

class PRRequest(BaseModel):
    """Input model for PR analysis request"""
    repo_url: str
    pr_number: int
    github_token: Optional[str] = None

class FileIssue(BaseModel):
    """Represents a single issue found in a file"""
    description: str = Field(description="Description of the issue")
    suggestion: str = Field(description="Suggestion to fix the issue")
    line: int = Field(description="Line number where the issue occurs")
    type: Literal["bug", "security", "perf"] = Field(description="Type of code issue")

class FileAnalysis(BaseModel):
    """Analysis result for a single file"""
    name: str
    issues: List[FileIssue] = Field(default_factory=list)

# Tools
async def fetch_pr_files(repo_url: str, pr_number: int, token: Optional[str] = None):
    """Fetch files from a GitHub PR asynchronously"""
    try:
        # Note: Github API itself is not async, but we can make the function async for the workflow
        g = Github(token) if token else Github()
        repo = g.get_repo("/".join(repo_url.split("/")[-2:]))
        pr = repo.get_pull(pr_number)
        
        return list(pr.get_files())
    except Exception as e:
        raise Exception(f"Error fetching PR files: {str(e)}")

async def fetch_file_content(raw_url: str, token: Optional[str] = None):
    """Fetch the full content of a file from its raw URL asynchronously"""
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"token {token}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(raw_url, headers=headers) as response:
                response.raise_for_status()
                return await response.text()
    except Exception as e:
        raise Exception(f"Error fetching file content: {str(e)}")

# Node Functions
def create_code_review_workflow():
    """Create the LangGraph workflow for code review"""
    # Initialize LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    async def fetch_files(state: CodeReviewState) -> CodeReviewState:
        """Fetch files from the PR"""
        try:
            files = await fetch_pr_files(
                state["repo_url"], 
                state["pr_number"], 
                state.get("github_token")
            )
            state["files"] = files
        except Exception as e:
            state["error"] = str(e)
            state["status"] = "failed"
            state["completed_at"] = datetime.utcnow().isoformat()            
        
        return state
    
    async def fetch_all_file_contents(state: CodeReviewState) -> CodeReviewState:
        """Fetch the full content of all files in the PR"""
        try:
            file_contents = {}
            files = state["files"]
            token = state.get("github_token")
            
            # Process all files
            for file in files:
                # Skip files that were deleted in the PR
                if file.status == "removed":
                    file_contents[file.filename] = ""
                    continue
                
                # Fetch the full content of the file
                content = await fetch_file_content(file.raw_url, token)
                file_contents[file.filename] = content
            
            state["file_contents"] = file_contents
        except Exception as e:
            state["status"] = "failed"
            state["completed_at"] = datetime.utcnow().isoformat()
            state["error"] = f"Error fetching file contents: {str(e)}"
            
        return state
    
    async def analyze_files(state: CodeReviewState) -> CodeReviewState:
        """Analyze each file in the PR with both full content and changes"""
        # Create the prompt for comprehensive code analysis
        system_message = SystemMessage(content="""
            You are an expert code reviewer. You have access to both the full file content and the specific changes made in a pull request.

            For your analysis, focus on:
            1. Potential bugs or errors
            2. Performance improvements
            3. Security concerns
            4. Context-aware issues (how changes affect the rest of the file)
            
            The changes are marked with a "+" for additions and "-" for deletions in the patch.
            Analyze both the specific changes and how they integrate with the full file.
                                       
            Types of issues:
            - "bug": Potential errors or logical issues
            - "security": Security vulnerabilities
            - "perf": Performance concerns
                                       
            Provide specific, actionable, and structured feedback
          """)
                  
        analyzed_files = []
        files = state["files"]
        file_contents = state["file_contents"]
        summary = {"total_files": 0, "total_issues": 0, "critical_issues": 0}
        
        for file in files:
            # Skip deleted files
            if file.status == "removed":
                continue
            
            full_content = file_contents.get(file.filename, "")
            user_message = HumanMessage(content=f"""Full File Content:
            {full_content}
            
            Changes in this PR:
            {file.patch}
            """)
        
            analyzer = llm.with_structured_output(FileAnalysis)
            analysis = await analyzer.ainvoke([system_message, user_message])
            analysis_dict = analysis.model_dump()

            issues = analysis_dict.get("issues", [])
            analyzed_files.append({
                "name": file.filename,
                "issues": issues
            })
            
            # Update summary
            summary["total_files"] += 1
            summary["total_issues"] += len(issues)
            summary["critical_issues"] += len([
                issue for issue in issues 
                if issue.get("type") in ["bug", "security"]
            ])
         
        state["summary"] = summary
        state["analyzed_files"] = analyzed_files
        
        return state
    
    async def finalize_review(state: CodeReviewState) -> CodeReviewState:
        """Finalize the review process"""
        state["completed_at"] = datetime.utcnow().isoformat()
        state["status"] = "completed" if not state.get("error") else "failed"

        return state     
    
    # Error detection function
    def check_for_errors(state: CodeReviewState):
        """Check if there are any errors in the state"""
        return "error" if state.get("error") or state["status"] == "failed" else "continue"
    
    # Workflow Graph
    workflow = StateGraph(CodeReviewState)
        
    # Adding nodes
    workflow.add_node("fetch_files", fetch_files)
    workflow.add_node("fetch_file_contents", fetch_all_file_contents)
    workflow.add_node("analyze_files", analyze_files)
    workflow.add_node("finalize", finalize_review)

    # Adding edges
    workflow.add_edge(START, "fetch_files")
    workflow.add_conditional_edges(
        "fetch_files",
        check_for_errors,
        {"continue": "fetch_file_contents", "error": "finalize"}
    )
    workflow.add_conditional_edges(
        "fetch_file_contents",
        check_for_errors,
        {"continue": "analyze_files", "error": "finalize"}
    )
    workflow.add_edge("analyze_files", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()

# Main Function to Run the Agent
async def analyze_pull_request(repo_url: str, pr_number: int, github_token: Optional[str] = None) -> Dict:
    """Analyze a GitHub Pull Request"""
    try:
        request = PRRequest(repo_url=repo_url, pr_number=pr_number, github_token=github_token)
    except ValueError as e:
        return {"status": "failed", "error": str(e)}

    initial_state: CodeReviewState = {
        "status": "pending",
        "repo_url": request.repo_url,
        "pr_number": request.pr_number,
        "github_token": request.github_token,
        "files": [],
        "started_at": datetime.utcnow().isoformat(),
        "error": None,
        "summary": {"total_files": 0, "total_issues": 0, "critical_issues": 0},
        "analyzed_files": [],
        "file_contents": {},
        "completed_at": None
    }

    workflow = create_code_review_workflow()
    result = await workflow.ainvoke(initial_state)

    return {"summary": result["summary"], "files": result["analyzed_files"]}