import os
import openai
import time
from datetime import datetime
from anthropic import Anthropic
from google.generativeai import GenerativeModel
from mistralai.client import MistralClient
from dotenv import load_dotenv

load_dotenv()

# API Clients
openai.api_key = os.getenv("OPENAI_API_KEY")
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
mistral_api_key = os.getenv("MISTRAL_API_KEY")

# OpenAI Client (>=1.0 SDK)
from openai import OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Token Estimation Utility
try:
    import tiktoken

    def estimate_tokens(model: str, prompt: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")  # fallback
        return len(encoding.encode(prompt))
except ImportError:
    def estimate_tokens(model: str, prompt: str) -> int:
        return 0  # Safe fallback if tiktoken is not available

# ======================
# Provider Connectors
# ======================

def chat_with_openai(prompt, settings):
    start_time = time.time()
    model = settings.get("model", "gpt-4o")
    stream = settings.get("stream", False)

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": settings.get("custom_instructions", "")},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.get("temperature", 0.7),
        max_tokens=settings.get("max_tokens", 1000),
        top_p=settings.get("top_p", 1),
        stream=stream
    )

    latency = int((time.time() - start_time) * 1000)
    content = "".join([chunk.choices[0].delta.content or "" for chunk in response]) if stream else response.choices[0].message.content
    tokens = None if stream else response.usage.total_tokens
    if tokens is None:
        tokens = estimate_tokens(model, prompt)

    return {
        "provider": "openai",
        "response": content,
        "tokens": tokens,
        "model": model,
        "latency": latency,
        "timestamp": datetime.utcnow().isoformat()
    }

def chat_with_claude(prompt, settings):
    start_time = time.time()
    model = settings.get("model", "claude-3-haiku-20240307")
    full_prompt = f"{settings.get('custom_instructions', '')}\n\n{prompt}"

    response = anthropic_client.messages.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=settings.get("max_tokens", 1000),
        temperature=settings.get("temperature", 0.7),
        top_p=settings.get("top_p", 1)
    )

    latency = int((time.time() - start_time) * 1000)
    total_tokens = response.usage.input_tokens + response.usage.output_tokens

    return {
        "provider": "claude",
        "response": response.content[0].text,
        "tokens": total_tokens,
        "model": model,
        "latency": latency,
        "timestamp": datetime.utcnow().isoformat()
    }

def chat_with_gemini(prompt, settings):
    start_time = time.time()
    model_id = settings.get("model", "gemini-1.5-flash-latest")
    model = GenerativeModel(model_id)
    response = model.generate_content(prompt)
    latency = int((time.time() - start_time) * 1000)
    tokens = estimate_tokens("gpt-4", prompt)  # estimate based on similar encoding

    return {
        "provider": "gemini",
        "response": response.text,
        "tokens": tokens,
        "model": model_id,
        "latency": latency,
        "timestamp": datetime.utcnow().isoformat()
    }

def chat_with_mistral(prompt, settings):
    start_time = time.time()
    model_id = settings.get("model", "mistral-medium")
    client = MistralClient(api_key=mistral_api_key)
    response = client.chat(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.get("temperature", 0.7),
        max_tokens=settings.get("max_tokens", 1000),
        top_p=settings.get("top_p", 1)
    )
    latency = int((time.time() - start_time) * 1000)

    try:
        tokens = response.usage.total_tokens
    except AttributeError:
        tokens = estimate_tokens("gpt-4", prompt)

    return {
        "provider": "mistral",
        "response": response.choices[0].message.content,
        "tokens": tokens,
        "model": model_id,
        "latency": latency,
        "timestamp": datetime.utcnow().isoformat()
    }

def chat_with_deepseek(prompt, settings):
    return {
        "provider": "deepseek",
        "response": "DeepSeek integration is pending or private API.",
        "tokens": estimate_tokens("gpt-4", prompt),
        "model": settings.get("model", "deepseek"),
        "latency": 0,
        "timestamp": datetime.utcnow().isoformat()
    }

def chat_with_grok(prompt, settings):
    return {
        "provider": "grok",
        "response": "Grok API is not publicly available.",
        "tokens": estimate_tokens("gpt-4", prompt),
        "model": settings.get("model", "grok"),
        "latency": 0,
        "timestamp": datetime.utcnow().isoformat()
    }

# ========================
# Unified Router Function
# ========================

def llm_router(provider, prompt, settings):
    if provider == "openai":
        return chat_with_openai(prompt, settings)
    elif provider == "claude":
        return chat_with_claude(prompt, settings)
    elif provider == "gemini":
        return chat_with_gemini(prompt, settings)
    elif provider == "mistral":
        return chat_with_mistral(prompt, settings)
    elif provider == "deepseek":
        return chat_with_deepseek(prompt, settings)
    elif provider == "grok":
        return chat_with_grok(prompt, settings)
    else:
        return {
            "error": "Unsupported provider",
            "timestamp": datetime.utcnow().isoformat()
        }
