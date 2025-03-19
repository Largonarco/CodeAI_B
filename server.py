import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# Import Firebase and MongodDB
import firebase_admin
from pymongo import AsyncMongoClient
from firebase_admin import credentials, auth
# Import your multi-agent workflow
from agents.supervisor import run_multi_agent_workflow

# Load ENV variables
load_dotenv()

# MongoDB URI
async_mongo_client = None
MONGODB_URI = os.getenv("MONGODB_URI")
# Firebase Creds
current_dir = os.path.dirname(os.path.abspath(__file__))
creds_path = os.path.join(current_dir, "credentials.json")

# Initialize MongoDB
try:
    async_mongo_client = AsyncMongoClient(MONGODB_URI)
    print("MongoDB initialized successfully")
except Exception as e:
    print(f"Error initializing MongoDB: {e}")
    raise

# Initialize Firebase Admin
try:
    cred = credentials.Certificate(creds_path)
    firebase_app = firebase_admin.initialize_app(cred)
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    raise


# FastAPI app
app = FastAPI(title="CodeAI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Security bearer token for Firebase JWT
security = HTTPBearer()

# Models
class UserCreateRequest(BaseModel):
    uid: str
    email: str
    display_name: Optional[str] = None
    profile_picture: Optional[str] = None

class AgentRequest(BaseModel):
    request: str
    github_token: Optional[str] = None

class MessageResponse(BaseModel):
    messages: List[Dict]
    results: Optional[Dict] = None

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, any]:
    """
    Verify Firebase JWT token and return user info.
    This function is used as a dependency for protected routes.
    
    Args:
        credentials: Bearer token credentials
        
    Returns:
        Dict containing user information from the decoded token
        
    Raises:
        HTTPException: If the token is invalid or missing
    """
    token = credentials.credentials
    try:
        # Verify the token with Firebase Admin SDK
        decoded_token = auth.verify_id_token(token)
        
        # Return the decoded token information
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}"
        )

# Routes
@app.post("/agent/run")
async def run_agent(agent_request: AgentRequest, user=Depends(get_current_user)):
    try:
        # Config for persistence across API calls on user basis
        config = {"configurable": {"thread_id": user.get("uid")}}
        # Run the multi-agent workflow
        result = await run_multi_agent_workflow(
            config=config,
            request=agent_request.request,
            async_mongo_client=async_mongo_client,
            github_token=agent_request.github_token
        )

        return result
    except Exception as e:
        # Log the error
        print(f"Error running agent workflow: {str(e)}")
        
        # Return appropriate error response
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error running agent workflow: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
