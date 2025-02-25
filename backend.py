from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
import logging
import re
from typing import Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Advanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("gorgeous_chatbot_backend")

app = FastAPI(title="Gorgeous Chatbot Generator API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify only your extension's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

class RequestData(BaseModel):
    message: str
    url: HttpUrl

    @field_validator('message', mode="before")
    def non_empty_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty")
        return value

class ResponseData(BaseModel):
    answer: str

def extract_core_domain(url: str) -> str:
    pattern = re.compile(r"^(https?://[^/]+)")
    match = pattern.match(url)
    if match:
        return match.group(1)
    raise ValueError("Invalid URL provided")

@app.post("/generate", response_model=ResponseData)
async def generate_response(data: RequestData) -> Any:
    try:
        core_domain = extract_core_domain(data.url)
    except ValueError as e:
        logger.error(f"Error extracting core domain: {e}")
        raise HTTPException(status_code=400, detail="Invalid URL format")

    logger.info(f"Received message: '{data.message}' from domain: {core_domain}")

    try:
        # Replace this simulated logic with your actual generation function.
        generated_answer = f"Processed message '{data.message}' from {core_domain}."
    except Exception as e:
        logger.exception("Processing error")
        raise HTTPException(status_code=500, detail="Generation error")

    return ResponseData(answer=generated_answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
