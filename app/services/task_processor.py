import asyncio
import litellm
from app.services.memory_manager import read_knowledge_base
from app.services.job_manager import update_job, add_message_to_history, get_history
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt
from app.services.usage_tracker import record_llm_usage

async def _run_llm_task_async(job_id: str, task_type: str, text: str, session_id: str = "default_session"):
    try:
        history = await asyncio.to_thread(get_history, session_id)
        context = await asyncio.to_thread(read_knowledge_base, text)
        model = select_model(task_type, text)
        system_prompt = get_system_prompt(task_type, context=context)

        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": text}]

        response = await litellm.acompletion(
            model=model,
            messages=messages,
        )

        usage = getattr(response, "usage", None)
        if usage:
            pt = int(getattr(usage, "prompt_tokens", 0))
            ct = int(getattr(usage, "completion_tokens", 0))
            if pt or ct:
                record_llm_usage(model=model, prompt_tokens=pt, completion_tokens=ct)

        result = response.choices[0].message.content
        add_message_to_history(session_id, "user", text)
        add_message_to_history(session_id, "assistant", result)

        update_job(job_id, "completed", {"response": result, "model_used": model})

    except Exception as e:
        update_job(job_id, "failed", str(e))


def run_llm_task(job_id: str, task_type: str, text: str, session_id: str = "default_session"):
    asyncio.run(_run_llm_task_async(job_id, task_type, text, session_id))
