"""
FastAPI Backend Server with WebSocket for real-time ML streaming
Connects ML pipeline to frontend dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import cv2
import numpy as np
import base64
import json
from typing import Dict, List
import sys
import os
import logging
import requests

# Add ml directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml', 'src'))

from inference import NeuroFlowPipeline
from rag_retriever import generate_intervention_text

app = FastAPI(title="NeuroFlow Backend API")

# CORS middleware
origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allow_origins = ["*"] if origins_env.strip() == "*" else [o.strip() for o in origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
ml_pipeline = None
# Optional Hugging Face text generator (lazy)
hf_text_generator = None

# HF Inference API configuration
HF_API_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_API_TOKEN")
HF_INFERENCE_MODEL = os.getenv("HF_INFERENCE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_INFERENCE_MODEL}"
USE_HF_INFERENCE = os.getenv("HF_USE_INFERENCE_API", "true").lower() in ("1", "true", "yes")

# Active WebSocket connections
active_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup_event():
    """Initialize ML pipeline on startup"""
    global ml_pipeline
    print("🚀 Starting NeuroFlow Backend...")
    ml_pipeline = NeuroFlowPipeline()
    print("✓ ML Pipeline Ready")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "NeuroFlow Backend",
        "status": "running",
        "ml_pipeline": "ready" if ml_pipeline else "not initialized"
    }


@app.get("/api/interventions")
async def get_interventions():
    """Get all available interventions"""
    if not ml_pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)
    
    return {
        "interventions": ml_pipeline.intervention_retriever.interventions
    }


@app.get("/api/session-summary")
async def get_session_summary():
    """Get current session summary"""
    if not ml_pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)
    
    summary = ml_pipeline.get_session_summary()
    return summary


@app.get("/api/state")
async def get_state():
    """Return current ML state (focus, stress, etc.)."""
    if not ml_pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)
    return ml_pipeline.current_state


@app.post("/api/intervention-feedback")
async def record_intervention_feedback(feedback: Dict):
    """
    Record user feedback on intervention
    
    Expected payload:
    {
        "user_response": "followed" | "accepted" | "dismissed" | "ignored",
        "state_before": {...},
        "state_after": {...},
        "action": 0-3
    }
    """
    if not ml_pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)
    
    try:
        ml_pipeline.record_intervention_outcome(
            user_response=feedback['user_response'],
            state_before=feedback['state_before'],
            state_after=feedback['state_after'],
            action=feedback['action']
        )
        
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


def _ensure_hf_generator():
    """Lazy-load a small Hugging Face text-generation pipeline.
    Falls back to None if transformers unavailable or model fails to load.
    """
    global hf_text_generator
    if hf_text_generator is not None:
        return hf_text_generator
    try:
        from transformers import pipeline
        # Use a tiny model by default; users can swap to a larger model via HF_TEXT_MODEL env.
        model_name = os.getenv("HF_TEXT_MODEL", "sshleifer/tiny-gpt2")
        hf_text_generator = pipeline("text-generation", model=model_name)
        logging.info(f"Initialized HF text-generation model: {model_name}")
    except Exception as e:
        logging.warning(f"Hugging Face pipeline unavailable, using template generation. Error: {e}")
        hf_text_generator = None
    return hf_text_generator


def _hf_infer(prompt: str, max_new_tokens: int = 128) -> str:
    """Call Hugging Face Inference API for text generation.
    Returns generated text or raises an Exception on failure.
    """
    if not USE_HF_INFERENCE or not HF_API_TOKEN:
        raise RuntimeError("HF Inference API disabled or missing token")
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.2,
            "top_p": 0.9,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True}
    }
    resp = requests.post(HF_INFERENCE_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HF API {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if isinstance(data, list) and data and isinstance(data[0], dict):
        txt = data[0].get("generated_text") or data[0].get("summary_text") or ""
        return txt.strip()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"HF API error: {data.get('error')}")
    raise RuntimeError("HF API: unexpected response format")


@app.get("/api/recommendations")
async def api_recommendations(top_k: int = 3):
    """Generate recommendations text based on current state using RAG + optional LLM."""
    if not ml_pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)

    # Retrieve candidate interventions using RAG
    state = ml_pipeline.current_state
    try:
        candidates = ml_pipeline.intervention_retriever.retrieve_interventions(state, top_k=top_k)
    except Exception as e:
        return JSONResponse({"error": f"RAG retrieval failed: {e}"}, status_code=500)

    # Try to craft a concise recommendation using HF Inference API; fallback to local transformers; then template
    gen = _ensure_hf_generator()
    results = []
    for item in candidates:
        text = None
        # 1) HF Inference API (preferred)
        if not text:
            try:
                prompt = (
                    f"You are a helpful focus and wellbeing coach. Based on the user's state: "
                    f"stress={state.get('stress', 0):.2f}, focus={state.get('focus', 0.5):.2f}. "
                    f"Recommend the intervention '{item['name']}' and explain why briefly. "
                    f"Give clear, numbered instructions. Keep under 80 words."
                )
                text = _hf_infer(prompt, max_new_tokens=120)
                if not text or text.lower().startswith('you are a helpful') or text.count('factors') > 5:
                    text = None
            except Exception as e:
                logging.info(f"HF inference recommendation failed: {e}")
        # 2) Local transformers pipeline (if available)
        if not text and gen is not None:
            try:
                prompt = (
                    f"You are a helpful focus and wellbeing coach. Based on the user's state: "
                    f"stress={state.get('stress', 0):.2f}, focus={state.get('focus', 0.5):.2f}. "
                    f"Recommend the intervention '{item['name']}' and explain why briefly. "
                    f"Give clear, numbered instructions. Keep under 80 words."
                )
                out = gen(prompt, max_new_tokens=96, do_sample=False, return_full_text=False)
                text = (out[0].get('generated_text', '') or out[0].get('summary_text', '')).strip()
                # Basic cleanup: strip prompt echoes or nonsense
                if not text or text.lower().startswith('you are a helpful') or text.count('factors') > 5:
                    text = None
            except Exception as e:
                logging.warning(f"HF generation failed, falling back to template: {e}")
        if not text:
            text = generate_intervention_text(item, llm_model=None)
        results.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "category": item.get("category"),
            "duration_seconds": item.get("duration_seconds"),
            "score": item.get("retrieval_score"),
            "text": text,
        })

    return {"state": state, "recommendations": results}


@app.post("/api/chat")
async def api_chat(payload: Dict = Body(...)):
    """Simple RAG-backed chat about focus/stress interventions.
    payload: { "question": str }
    """
    if not ml_pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)

    question = (payload or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    state = ml_pipeline.current_state
    retriever = ml_pipeline.intervention_retriever
    try:
        docs = retriever.retrieve_interventions(state, top_k=5)
    except Exception as e:
        return JSONResponse({"error": f"retrieval failed: {e}"}, status_code=500)

    # Build a compact context block from top docs
    context_lines = []
    for d in docs:
        context_lines.append(f"- {d['name']}: {d['description']} Instructions: {d['instructions']}")
    context = "\n".join(context_lines[:5])

    # Try HF Inference API first, then local transformers; fallback to extractive answer
    gen = _ensure_hf_generator()
    if gen is not None or (USE_HF_INFERENCE and HF_API_TOKEN):
        try:
            prompt = (
                "You are a concise coach. Use the context to answer the user's question.\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n"
                "Answer in 2-5 sentences with specific, practical steps."
            )
            answer = None
            # 1) HF Inference API
            if USE_HF_INFERENCE and HF_API_TOKEN:
                try:
                    answer = _hf_infer(prompt, max_new_tokens=160)
                except Exception as e:
                    logging.info(f"HF inference chat failed: {e}")
            # 2) Local transformers
            if (not answer) and gen is not None:
                out = gen(prompt, max_new_tokens=160, do_sample=False, return_full_text=False)
                answer = (out[0].get('generated_text', '') or out[0].get('summary_text', '')).strip()
            if (not answer) or answer.lower().startswith('you are a concise') or answer.count('factors') > 5:
                raise RuntimeError('LLM output low quality; using fallback')
            return {"answer": answer, "used_llm": True}
        except Exception as e:
            logging.warning(f"HF chat generation failed, using fallback: {e}")

    # Fallback: Select the most relevant doc by simple keyword overlap
    q = question.lower()
    scored = []
    for d in docs:
        score = sum(w in (d['name'] + ' ' + d['description'] + ' ' + d['instructions']).lower() for w in q.split())
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[0][1] if scored else (docs[0] if docs else None)
    if not top:
        return {"answer": "I couldn't find relevant guidance right now. Try rephrasing your question.", "used_llm": False}
    answer = (
        f"Based on your question, {top['name']} can help. "
        f"Why: {top['description']}. Try this: {top['instructions']}"
    )
    return {"answer": answer, "used_llm": False}


@app.websocket("/ws/neuroflow")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time ML processing
    
    Protocol:
    - Client sends: {"type": "frame", "image": "base64...", "keystroke": bool, "cursor": [x, y]}
    - Server sends: {"type": "analysis", "data": {...}}
    """
    await websocket.accept()
    active_connections.append(websocket)
    
    print(f"✓ Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            if data['type'] == 'frame':
                # Decode base64 image
                image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Process frame through ML pipeline
                    keystroke = data.get('keystroke', False)
                    cursor_pos = tuple(data.get('cursor', [0, 0]))
                    
                    result = ml_pipeline.process_frame(frame, keystroke, cursor_pos)
                    
                    # Send result back to client
                    await websocket.send_json({
                        "type": "analysis",
                        "data": result
                    })
            
            elif data['type'] == 'ping':
                # Keep-alive
                await websocket.send_json({"type": "pong"})
            
            elif data['type'] == 'get_state':
                # Send current state
                await websocket.send_json({
                    "type": "state",
                    "data": ml_pipeline.current_state
                })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"✗ Client disconnected. Total connections: {len(active_connections)}")
    
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        active_connections.remove(websocket)


@app.post("/api/upload-frame")
async def upload_frame(
    file: UploadFile = File(...),
    keystroke: bool = False,
    cursor_x: int = 0,
    cursor_y: int = 0
):
    """
    Alternative REST endpoint for frame processing
    Useful for testing without WebSocket
    """
    if not ml_pipeline:
        return JSONResponse({"error": "Pipeline not initialized"}, status_code=500)
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)
        
        # Process
        result = ml_pipeline.process_frame(
            frame,
            keystroke=keystroke,
            cursor_pos=(cursor_x, cursor_y)
        )
        
        return result
    
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == '__main__':
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║         🧠 NeuroFlow Backend Server                  ║
    ║                                                       ║
    ║  Real-time Focus & Stress Analysis API               ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )