"""OpenAI conversation transport; vision and robot tools remain local."""
import inspect
import json
import os
from types import SimpleNamespace

import requests


def runtime_identity(backend, model, fallback=False):
    """Describe the actual response path, including command-line overrides."""
    location = "OpenAI API" if backend == "openai" else (
        "local Transformers Qwen-VL" if backend == "transformers" else "Ollama")
    return (
        " Current authoritative runtime identity for this reply: "
        + json.dumps({"backend": backend, "model_id": model,
                      "runtime": location, "local_fallback": fallback})
        + ". When asked which model or provider you use, report this exact model_id "
        "and backend. This overrides identity claims from training or earlier replies. "
        "It is the configured model identifier, not a verified underlying model "
        "snapshot; do not invent version numbers. Keep your name Fluffy. "
        "Do not search the web for your runtime identity."
    )


def require_api_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OpenAI conversation requires OPENAI_API_KEY in the environment")
    return key


def chat(model, messages, tools=None):
    """Return the same small response interface used by the Ollama tool loop."""
    schemas = []
    for function in tools or []:
        parameters = inspect.signature(function).parameters
        schemas.append({
            "type": "function",
            "function": {
                "name": function.__name__,
                "description": inspect.getdoc(function) or function.__name__,
                "parameters": {
                    "type": "object",
                    "properties": {name: {"type": "string"} for name in parameters},
                    "required": list(parameters),
                    "additionalProperties": False,
                },
            },
        })
    payload = {
        "model": model,
        "messages": messages,
        "reasoning_effort": "none",
        "service_tier": "fast",
        "max_completion_tokens": 512,
    }
    if schemas:
        payload["tools"] = schemas
        payload["parallel_tool_calls"] = False
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {require_api_key()}"},
        json=payload,
        timeout=(10, 60),
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    calls = []
    for call in message.get("tool_calls") or []:
        arguments = json.loads(call["function"]["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("OpenAI tool arguments must be an object")
        calls.append(SimpleNamespace(
            id=call["id"],
            function=SimpleNamespace(name=call["function"]["name"], arguments=arguments),
        ))
    return SimpleNamespace(
        message=SimpleNamespace(content=message.get("content"), tool_calls=calls),
        raw_message=message,
    )
