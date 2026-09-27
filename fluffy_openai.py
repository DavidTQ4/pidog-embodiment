"""OpenAI conversation transport; vision and robot tools remain local."""
import inspect
import json
import os
from types import SimpleNamespace

import requests


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
