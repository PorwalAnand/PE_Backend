def calculate_cost(model: str, tokens: int) -> float:
    """
    Calculate estimated cost for a given model and number of tokens.
    Prices are per 1,000 tokens and may vary per model.
    """
    if tokens is None or tokens == 0:
        return 0.0

    price_per_1k_tokens = {
        # OpenAI
        "gpt-4o": 0.005,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.002,

        # Claude
        "claude-3-haiku-20240307": 0.002,

        # Gemini
        "gemini-1.5-flash-latest": 0.0015,

        # DeepSeek (placeholder)
        "deepseek-coder": 0.001,

        # Mistral
        "mistral-7b-instruct": 0.0008,
    }

    price_per_token = price_per_1k_tokens.get(model, 0)
    cost = (tokens / 1000) * price_per_token
    return round(cost, 6)
