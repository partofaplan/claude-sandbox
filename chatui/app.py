import json
import os

import requests
from flask import Flask, Response, render_template, request, stream_with_context

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


@app.route("/")
def index():
    return render_template("index.html", model=MODEL)


@app.route("/health")
def health():
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_status = "ok" if resp.status_code == 200 else "unavailable"
    except Exception:
        ollama_status = "unavailable"
    return {"status": "ok", "ollama": ollama_status, "model": MODEL}


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

    if not messages:
        return {"error": "No messages provided"}, 400

    def generate():
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": MODEL, "messages": messages, "stream": True},
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        content = chunk["message"].get("content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    if chunk.get("done"):
                        yield "data: [DONE]\n\n"
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'error': 'Cannot connect to Ollama'})}\n\n"
            yield "data: [DONE]\n\n"
        except requests.exceptions.HTTPError as e:
            yield f"data: {json.dumps({'error': f'Ollama error: {e.response.status_code}'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
