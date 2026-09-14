import json
import os
from urllib.request import Request, urlopen

def _load_local_env():
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    try:
        with open(env_file, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip("\"'"))
    except OSError:
        pass

def general_answer(question, language="en"):
    _load_local_env()
    if str(os.getenv("CROWDGUARD_LLM_ENABLED", "false")).lower() != "true":
        return None
    provider = os.getenv("CROWDGUARD_LLM_PROVIDER", "openai").lower()
    key = os.getenv("CROWDGUARD_LLM_API_KEY")
    if not key:
        return None
    if provider == "openrouter":
        body = {"model": os.getenv("CROWDGUARD_LLM_MODEL", "openrouter/free"), "messages": [{"role": "system", "content": "Answer general questions briefly. Do not claim access to CrowdGuard live metrics."}, {"role": "user", "content": question}], "temperature": 0.2, "max_tokens": 300}
        try:
            request = Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "HTTP-Referer": "http://localhost:5173", "X-Title": "CrowdGuard Assistant"}, method="POST")
            with urlopen(request, timeout=15) as response:
                return json.loads(response.read()).get("choices", [{}])[0].get("message", {}).get("content")
        except Exception:
            return None
    if provider == "gemini":
        model = os.getenv("CROWDGUARD_LLM_MODEL", "gemini-3.6-flash")
        body = {"contents": [{"parts": [{"text": "Answer general questions briefly. Do not claim access to CrowdGuard live metrics.\n\nQuestion: " + question}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}}
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + key
            request = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=12) as response:
                data = json.loads(response.read())
                return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")
        except Exception:
            return None
    if provider != "openai":
        return None
    body = {"model": os.getenv("CROWDGUARD_LLM_MODEL", "gpt-4o-mini"), "messages": [{"role": "system", "content": "Answer general questions briefly. Do not claim access to CrowdGuard live metrics."}, {"role": "user", "content": question}], "temperature": 0.2, "max_tokens": 300}
    try:
        request = Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=12) as response:
            return json.loads(response.read()).get("choices", [{}])[0].get("message", {}).get("content")
    except Exception:
        return None
