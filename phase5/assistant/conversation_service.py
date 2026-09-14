from .deterministic_answers import answer_question
from .llm_adapter import general_answer

def chat(question, context, conversation=None, language="auto"):
    if any(token in question.lower() for token in ("api key", "password", "smtp", ".env", "app password", "secret", "token")):
        return {"answer": "I can’t provide passwords, API keys, tokens, or private configuration.", "intent": "UNKNOWN", "language": "en", "mode": "SAFETY_REFUSAL", "evidence": {}, "used_sources": [], "suggested_followups": ["Ask about the current crowd", "Ask about the forecast"]}
    result = answer_question(question, context, conversation, language)
    if result["intent"] == "GENERAL_CONVERSATION":
        generated = general_answer(question, result["language"])
        if generated:
            result.update(answer=generated, mode="GENERAL_AI", used_sources=["optional LLM"])
        else:
            result["answer"] = "General AI response is temporarily unavailable. I can still answer CrowdGuard questions from the selected monitoring session."
    return result
