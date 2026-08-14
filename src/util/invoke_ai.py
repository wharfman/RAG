from openai import OpenAI

from src.config import Settings


def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke AI model given a system and user message
    Replace with whatever AI model you want
    """

    settings = Settings.from_env()
    client = OpenAI(api_key=settings.require_openai_key())
    response = client.responses.create(
        model=settings.generation_model,
        reasoning={"effort": "none"},
        instructions=system_message,
        input=user_message,
    )
    return response.output_text
