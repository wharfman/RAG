from src.interface.base_evaluator import BaseEvaluator, EvaluationResult
from src.util.extract_xml import extract_xml_tag
from src.util.invoke_ai import invoke_ai

SYSTEM_PROMPT = """
You are a system that evaluates the correctness of a response to a question.
The question will be provided in <question>...</question> tags.
The response will be provided in <response>...</response> tags.
The expected answer will be provided in <expected_answer>...</expected_answer> tags.

The response does not have to exactly match every word in the expected answer.
It only needs to correctly answer the actual question.

Evaluate whether the response is correct and return your reasoning in
<reasoning>...</reasoning> tags.
Then return the result in <result>...</result> tags — either as 'true' or 'false'.
"""


class Evaluator(BaseEvaluator):
    def evaluate(self, query: str, response: str, expected_answer: str) -> EvaluationResult:

        user_prompt = f"""
        <question>\n{query}\n</question>
        <response>\n{response}\n</response>
        <expected_answer>\n{expected_answer}\n</expected_answer>
        """

        response_content = invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_prompt)

        reasoning = extract_xml_tag(response_content, "reasoning")
        result = extract_xml_tag(response_content, "result")
        print(response_content)

        if result is not None:
            is_correct = result.lower() == "true"
        else:
            is_correct = False
            reasoning = "No result found"

        return EvaluationResult(
            question=query,
            response=response,
            expected_answer=expected_answer,
            is_correct=is_correct,
            reasoning=reasoning,
        )
