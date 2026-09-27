import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import fluffy_openai


def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        fluffy_openai.require_api_key()


def test_fast_request_and_tool_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sent = {}

    def post(url, **kwargs):
        sent.update(kwargs)
        assert url == "https://api.openai.com/v1/chat/completions"
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "choices": [{"message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "observe_scene", "arguments": "{}"}}]}}]})

    def observe_scene():
        """Read the current scene."""

    monkeypatch.setattr(fluffy_openai.requests, "post", post)
    response = fluffy_openai.chat("gpt-6-luna", [], [observe_scene])
    assert sent["json"]["service_tier"] == "fast"
    assert sent["json"]["reasoning_effort"] == "none"
    assert sent["json"]["tools"][0]["function"]["parameters"]["properties"] == {}
    assert response.message.tool_calls[0].id == "call_1"
    assert response.message.tool_calls[0].function.arguments == {}


def load_method(name, namespace):
    """Exercise production routing without loading CUDA/camera dependencies."""
    tree = ast.parse(Path("pidog_yolo_vlm.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLMObserver")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pidog_yolo_vlm.py", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("backend", ["openai", "ollama"])
def test_conversation_tool_round_trip(backend):
    import json
    import os

    turns = []
    observations = []

    def chat(*args, **kwargs):
        messages = args[1] if args else kwargs["messages"]
        turns.append(list(messages))
        if len(turns) == 1:
            message = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                id="call_scene", function=SimpleNamespace(name="observe_scene", arguments={}))])
            return SimpleNamespace(message=message, raw_message={"role": "assistant", "content": None})
        return SimpleNamespace(message=SimpleNamespace(content="I see a ball.", tool_calls=[]),
                               raw_message={"role": "assistant", "content": "I see a ball."})

    namespace = dict(ollama_chat=chat, openai_chat=chat, json=json, os=os,
                     sanitise_spoken_text=lambda text: text)
    method = load_method("_answer_with_conversation_model", namespace)
    observer = SimpleNamespace(conversation_backend=backend, conversation_model="model",
                               conversation_history=[], web_search_enabled=False)
    result = method(observer, "system", "{}", "What can you see?",
                    scene_observer=lambda: observations.append(True) or {"scene": "ball"})
    assert result == "I see a ball."
    assert observations == [True]
    tool_result = turns[1][-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id" if backend == "openai" else "tool_name"] == (
        "call_scene" if backend == "openai" else "observe_scene")


def test_cli_defaults_and_openai_selection(monkeypatch):
    import argparse
    tree = ast.parse(Path("pidog_yolo_vlm.py").read_text(encoding="utf-8"))
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_args")
    namespace = {n.id: 1 for n in ast.walk(method) if isinstance(n, ast.Name) and n.id.startswith("DEFAULT_")}
    namespace.update(argparse=argparse)
    namespace["DEFAULT_CONVERSATION_MODEL"] = "qwen3:8b"
    exec(compile(ast.Module(body=[method], type_ignores=[]), "cli", "exec"), namespace)
    monkeypatch.setattr("sys.argv", ["pidog_yolo_vlm.py"])
    args = namespace["parse_args"]()
    assert args.conversation_backend == "ollama"
    assert args.conversation_model == "qwen3:8b"
    monkeypatch.setattr("sys.argv", ["pidog_yolo_vlm.py", "--conversation-backend", "openai"])
    args = namespace["parse_args"]()
    assert args.openai_model == "gpt-6-luna"
    assert args.conversation_backend == "openai"

@pytest.mark.parametrize("backend,fail", [("openai", False), ("openai", True), ("ollama", False)])
def test_direct_camera_tool(backend, fail):
    import json
    import os
    turns, captures = [], []
    def capture():
        captures.append(True)
        if fail:
            raise RuntimeError("camera stalled")
        return "data:image/jpeg;base64,dGVzdA=="
    def chat(*args, **kwargs):
        messages = args[1] if args else kwargs["messages"]
        tools = args[2] if args else kwargs["tools"]
        names = [f.__name__ for f in tools or []]
        assert ("request_camera_still" in names) == (backend == "openai")
        turns.append(list(messages))
        calls = []
        if backend == "openai" and len(turns) < 3:
            calls = [SimpleNamespace(id="still", function=SimpleNamespace(name="request_camera_still", arguments={}))]
        return SimpleNamespace(message=SimpleNamespace(content="Answer", tool_calls=calls),
                               raw_message={"role": "assistant", "content": "Answer"})
    method = load_method("_answer_with_conversation_model", dict(
        ollama_chat=chat, openai_chat=chat, os=os, json=json, sanitise_spoken_text=lambda text: text))
    observer = SimpleNamespace(conversation_backend=backend, conversation_model="model",
                               conversation_history=[], web_search_enabled=False)
    assert method(observer, "system", "{}", "Read this label", camera_still_provider=capture) == "Answer"
    assert len(captures) == (1 if backend == "openai" else 0)
    if backend == "openai":
        images = [m for m in turns[-1] if isinstance(m.get("content"), list)]
        assert len(images) == (0 if fail else 1)
        if not fail:
            assert images[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert "Only one camera still" in turns[-1][-1]["content"]


def test_camera_encoding_and_stale_timeout(monkeypatch):
    import base64
    import cv2
    import numpy as np
    import fluffy_camera
    frame = np.zeros((1600, 2000, 3), dtype=np.uint8)
    samples = iter([(1, frame, None), (2, frame, None)])
    camera = SimpleNamespace(latest=lambda: next(samples))
    url = fluffy_camera.capture_camera_still(camera)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",")[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (1024, 1280)
    times = iter([0, 0, 3])
    monkeypatch.setattr(fluffy_camera.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(fluffy_camera.time, "sleep", lambda _: None)
    camera = SimpleNamespace(latest=lambda: (1, frame, None))
    with pytest.raises(RuntimeError, match="No fresh camera"):
        fluffy_camera.capture_camera_still(camera)
