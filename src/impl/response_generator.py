from typing import List
from interface.base_response_generator import BaseResponseGenerator


SYSTEM_PROMPT = """
Use the provided context to provide a concise answer to the user's question.
If you cannot find the answer in the context, say so. Do not make up information.
"""

class ResponseGenerator(BaseResponseGenerator):
    def generate_response(self, query, context) -> str:
        """Generate response using OpenAI's chat completion."""
        context_text = "\n".join(context)
        user_message = (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )

        return invoke_ai(
            system_message=SYSTEM_PROMPT,
            user_message=user_message
        )