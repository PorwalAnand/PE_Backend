import time
import uuid
import shutil
import os
from .database import db
from .models import Prompt
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request, Query, UploadFile, File
from pydantic import BaseModel
from app.llm_connectors import llm_router
from datetime import datetime
from app.models import ChatRequest, ChatResponse, ChatHistoryResponse, ChatRecord
from app.database import chat_history
from datetime import datetime
from app.database import model_usage
from fastapi.responses import JSONResponse
from collections import defaultdict
from app.utils import calculate_cost
from app.models import ModelComparisonRequest, ModelComparisonResponse, ComparisonResult, ModelSettings

router = APIRouter()
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class LLMRequest(BaseModel):
    provider: str
    prompt: str
    settings: dict

@router.get("/prompts", response_model=List[Prompt])
async def get_all_prompts():
    all_prompts = []
    skip_collections = {"chat_history", "model_usage"}  # do not list these

    for collection_name in db.list_collection_names():
        if collection_name in skip_collections:
            continue

        prompts = list(db[collection_name].find({}, {"_id": 0}))
        for prompt in prompts:
            # Add fallbacks to avoid validation error
            all_prompts.append({
                "promptName": prompt.get("promptName", ""),
                "promptDescription": prompt.get("promptDescription", ""),
                "promptCategory": prompt.get("promptCategory", collection_name),
            })

    return all_prompts



@router.get("/prompts/{category}", response_model=List[Prompt])
async def get_prompts_by_category(category: str):
    if category not in db.list_collection_names():
        return []
    prompts = list(db[category].find({}, {"_id": 0}))
    return prompts


@router.get("/analytics/usage")
async def usage_analytics(
    model: Optional[str] = Query("All"),
    startDate: Optional[str] = Query(None),
    endDate: Optional[str] = Query(None),
):
    match_conditions = {}

    # Filter by model if provided
    if model.lower() != "all":
        match_conditions["model"] = model

    # Filter by date range if provided
    date_filter = {}
    if startDate:
        date_filter["$gte"] = datetime.strptime(startDate, "%Y-%m-%d")
    if endDate:
        date_filter["$lte"] = datetime.strptime(endDate, "%Y-%m-%d")
    if date_filter:
        match_conditions["date"] = date_filter

    pipeline = []

    # Add $match only if any filters exist
    if match_conditions:
        pipeline.append({"$match": match_conditions})

    # Group by model and date
    pipeline.extend([
        {
            "$group": {
                "_id": {
                    "model": "$model",
                    "date": "$date"
                },
                "total_tokens": {"$sum": "$tokens"},
                "total_cost": {"$sum": "$cost"},
                "total_requests": {"$sum": 1}
            }
        },
        {
            "$sort": {"_id.date": 1}
        }
    ])

    usage_data = list(model_usage.aggregate(pipeline))

    # Format for frontend/charting
    results = defaultdict(lambda: {"dates": [], "tokens": [], "requests": [], "costs": []})
    for entry in usage_data:
        model_name = entry["_id"]["model"]
        date = entry["_id"]["date"]
        results[model_name]["dates"].append(date)
        results[model_name]["tokens"].append(entry["total_tokens"])
        results[model_name]["requests"].append(entry["total_requests"])
        results[model_name]["costs"].append(round(entry["total_cost"], 4))

    return JSONResponse(content=results)

@router.post("/llm/chat")
async def multi_llm_chat(request: LLMRequest):
    try:
        response = llm_router(request.provider, request.prompt, request.settings)

        session_id = request.settings.get("session_id") or str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Save chat history
        chat_history.insert_one({
            "model": request.provider,
            "prompt": request.prompt,
            "response": response.get("response") if isinstance(response, dict) else str(response),
            "timestamp": timestamp,
            "session_id": session_id
        })

        # Save model usage analytics (non-breaking)
        usage_data = response.get("usage", {}) if isinstance(response, dict) else {}
        tokens = usage_data.get("total_tokens", 0)

        if tokens > 0:
            # Define cost per token by provider (adjust rates as needed)
            cost_per_token = {
                "openai": 0.000002,     # $0.002 / 1K
                "claude": 0.000003,     # $0.003 / 1K
                "gemini": 0.0000015     # $0.0015 / 1K
            }

            cost = round(tokens * cost_per_token.get(request.provider, 0), 6)

        model_usage.insert_one({
            "model": request.settings.get("model", request.provider),
            "provider": request.provider,
            "tokens": response.get("tokens"),
            "cost": calculate_cost(response.get("model"), response.get("tokens")),
            "timestamp": datetime.utcnow(),
            "date": datetime.utcnow().strftime("%Y-%m-%d")
        })

        return {
            "provider": request.provider,
            "response": response,
            "timestamp": timestamp
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/chat", response_model=ChatResponse)
# async def chat_endpoint(payload: ChatRequest):
#     try:
#         response = llm_router(payload.provider.lower(), payload.prompt, payload.settings)

#         if isinstance(response, dict) and "message" in response:
#             content = response["message"]
#         elif hasattr(response, 'choices'):
#             content = response['choices'][0]['message']['content']
#         else:
#             content = response

#         # Save chat
#         session_id = payload.settings.get("session_id") or str(uuid.uuid4())
#         chat_history.insert_one({
#             "model": payload.provider,
#             "prompt": payload.prompt,
#             "response": content,
#             "timestamp": datetime.utcnow(),
#             "session_id": session_id
#         })

#         return ChatResponse(response=content)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/sessions")
async def list_chat_sessions():
    pipeline = [
        {
            "$group": {
                "_id": "$session_id",
                "model": {"$first": "$model"},
                "latest_prompt": {"$last": "$prompt"},
                "last_updated": {"$max": "$timestamp"}
            }
        },
        {"$sort": {"last_updated": -1}}
    ]
    sessions = list(chat_history.aggregate(pipeline))
    return [{"session_id": s["_id"], "model": s["model"], "last_updated": s["last_updated"], "latest_prompt": s["latest_prompt"]} for s in sessions]

    
@router.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(model: Optional[str] = None, session_id: Optional[str] = None):
    query = {}
    if model:
        query["model"] = model
    if session_id:
        query["session_id"] = session_id

    records = chat_history.find(query).sort("timestamp", -1)
    history = []
    for record in records:
        history.append(ChatRecord(
            model=record["model"],
            prompt=record["prompt"],
            response=record["response"],
            timestamp=record["timestamp"],
            session_id=record.get("session_id")
        ))
    return ChatHistoryResponse(history=history)

@router.post("/compare-models", response_model=ModelComparisonResponse)
async def compare_models(request: ModelComparisonRequest):
    prompt = request.prompt
    models = request.models
    settings_dict = request.settings
    files = request.files

    if not 1 <= len(models) <= 3:
        raise HTTPException(status_code=400, detail="Select 1 to 3 models only.")

    results = []

    for model in models:
        start = time.time()
        settings = settings_dict.get(model, ModelSettings()).dict()

        # Optional: Add file-handling logic here if needed

        # Append web search flag
        if settings.get("web_search"):
            prompt += " [Include web search results]"

        # Inject custom instruction
        if settings.get("custom_instructions"):
            prompt = f"{settings['custom_instructions']} \n\n{prompt}"

        try:
            response = llm_router(model, prompt, settings)

            if isinstance(response, dict):
                # Expected structure from llm_connector
                content = response.get("response", str(response))
                tokens = response.get("tokens", None)
            elif hasattr(response, "text"):
                # fallback case for models like Claude/Gemini returning text only
                content = response.text
                tokens = None
            else:
                content = str(response)
                tokens = None

        except Exception as e:
            content = f"Error: {str(e)}"
            tokens = None

        latency = round((time.time() - start) * 1000)

        results.append(ComparisonResult(
            provider=model,
            response=content,
            tokens=tokens,
            latency=latency
        ))

    return ModelComparisonResponse(comparisons=results)



@router.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"file_id": file_id, "filename": file.filename, "path": file_path}
