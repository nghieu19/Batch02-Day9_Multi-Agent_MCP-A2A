"""Shared LLM factory for all agents.

Priority:
1. GEMINI_API_KEY → dùng Google Gemini trực tiếp (free tier)
2. OPENROUTER_API_KEY → dùng OpenRouter (sk-or-...)
3. OPENAI_API_KEY → dùng OpenAI trực tiếp (sk-proj-...)
"""

import os


def get_llm():
    from dotenv import load_dotenv
    load_dotenv()

    gemini_key = os.getenv("GEMINI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # --- Ưu tiên 1: Gemini API (Google AI Studio, free) ---
    if gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_key,
            max_output_tokens=2048,
        )

    # --- Ưu tiên 2: OpenRouter ---
    if openrouter_key and not openrouter_key.startswith("sk-proj-"):
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")
        return ChatOpenAI(
            model=model_name,
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=2048,
        )

    # --- Ưu tiên 3: OpenAI trực tiếp ---
    api_key = openrouter_key or openai_key
    if api_key:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENROUTER_MODEL", "gpt-4o-mini")
        if model_name == "gpt-4.1-mini":
            model_name = "gpt-4o-mini"
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            max_tokens=2048,
        )

    raise ValueError(
        "Không tìm thấy API key! Vui lòng thêm GEMINI_API_KEY, OPENROUTER_API_KEY, hoặc OPENAI_API_KEY vào file .env"
    )