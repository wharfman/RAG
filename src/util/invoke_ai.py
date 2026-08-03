from openai import OpenAI

def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke AI model given a system and user message
    Replace with whatever AI model you want
    """

    client = OpenAI() # Insert API key here
    response = client.chat.completions.create(
        model = "gpt-5.4-mini",
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content