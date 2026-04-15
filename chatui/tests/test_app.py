import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    with patch("app.requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["ollama"] == "ok"
    assert "model" in data


def test_health_ollama_unavailable(client):
    with patch("app.requests.get", side_effect=Exception("timeout")):
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["ollama"] == "unavailable"


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def test_index_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Chat" in resp.data
    assert b"llama3.2" in resp.data


# ---------------------------------------------------------------------------
# Chat endpoint — input validation
# ---------------------------------------------------------------------------

def test_chat_no_body_returns_400(client):
    resp = client.post("/chat", content_type="application/json", data="{}")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_chat_empty_messages_returns_400(client):
    resp = client.post(
        "/chat",
        content_type="application/json",
        data=json.dumps({"messages": []}),
    )
    assert resp.status_code == 400


def test_chat_missing_messages_key_returns_400(client):
    resp = client.post(
        "/chat",
        content_type="application/json",
        data=json.dumps({"other": "stuff"}),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Chat endpoint — streaming success
# ---------------------------------------------------------------------------

def _make_sse_lines(*chunks, done=True):
    """Build a list of SSE byte lines the mock response will yield."""
    lines = []
    for content in chunks:
        payload = json.dumps({"message": {"role": "assistant", "content": content}, "done": False})
        lines.append(payload.encode())
    if done:
        lines.append(json.dumps({"done": True}).encode())
    return lines


def test_chat_streams_response(client):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = _make_sse_lines("Hello", " world")

    with patch("app.requests.post", return_value=mock_resp):
        resp = client.post(
            "/chat",
            content_type="application/json",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        )

    assert resp.status_code == 200
    assert resp.content_type.startswith("text/event-stream")
    body = resp.data.decode()
    assert "Hello" in body
    assert " world" in body
    assert "[DONE]" in body


def test_chat_sends_correct_payload_to_ollama(client):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = _make_sse_lines("ok")

    messages = [{"role": "user", "content": "ping"}]

    with patch("app.requests.post", return_value=mock_resp) as mock_post:
        client.post(
            "/chat",
            content_type="application/json",
            data=json.dumps({"messages": messages}),
        )

    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs.get("json") or call_kwargs.args[1] if len(call_kwargs.args) > 1 else call_kwargs.kwargs["json"]
    assert payload["messages"] == messages
    assert payload["stream"] is True
    assert "model" in payload


# ---------------------------------------------------------------------------
# Chat endpoint — error handling
# ---------------------------------------------------------------------------

def test_chat_connection_error_streams_error_event(client):
    import requests as req_lib

    with patch("app.requests.post", side_effect=req_lib.exceptions.ConnectionError("refused")):
        resp = client.post(
            "/chat",
            content_type="application/json",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        )

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "error" in body
    assert "Cannot connect to Ollama" in body
    assert "[DONE]" in body


def test_chat_http_error_streams_error_event(client):
    import requests as req_lib

    mock_err_resp = MagicMock()
    mock_err_resp.status_code = 503

    mock_post_resp = MagicMock()
    mock_post_resp.raise_for_status.side_effect = req_lib.exceptions.HTTPError(
        response=mock_err_resp
    )

    with patch("app.requests.post", return_value=mock_post_resp):
        resp = client.post(
            "/chat",
            content_type="application/json",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        )

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "error" in body
    assert "[DONE]" in body


def test_chat_generic_exception_streams_error_event(client):
    with patch("app.requests.post", side_effect=RuntimeError("boom")):
        resp = client.post(
            "/chat",
            content_type="application/json",
            data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        )

    assert resp.status_code == 200
    body = resp.data.decode()
    assert "boom" in body
    assert "[DONE]" in body
