import json
import litellm


JUDGE_PROMPT = """You are evaluating an AI assistant's response.
---
Task: {task}
User Input: {input}
Assistant Response: {response}
---
Criteria: {criteria}
---
Score this response on a scale of 1-5 where:
1 = Completely wrong, harmful, or off-topic
2 = Mostly wrong or misses key points
3 = Partially correct but has some issues
4 = Mostly correct with minor issues
5 = Perfectly accurate, complete, and well-structured

Return ONLY a JSON object with exactly these two fields:
{{"score": <int 1-5>, "reasoning": "<brief justification for the score>"}}
"""


async def judge_response(
    task: str,
    user_input: str,
    response: str,
    criteria: str,
    model: str = "openai/gpt-4o",
) -> tuple[int, str]:
    prompt = JUDGE_PROMPT.format(
        task=task,
        input=user_input,
        response=response,
        criteria=criteria,
    )
    try:
        result = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = result.choices[0].message.content
        if not content:
            return 0, "Judge returned empty response"
        parsed = json.loads(content)
        score = int(parsed.get("score", 0))
        reasoning = parsed.get("reasoning", "")
        return max(1, min(5, score)), reasoning
    except Exception as e:
        return 0, f"Judge error: {e}"
