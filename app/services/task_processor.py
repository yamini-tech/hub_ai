import litellm
from app.services.memory_manager import read_knowledge_base
from app.services.job_manager import update_job, add_message_to_history, get_history
from app.services.model_selector import select_model
from app.services.prompt_manager import get_system_prompt

def run_llm_task(job_id: str, task_type: str, text: str, session_id: str = "default_session"):
    try:
        history = get_history(session_id)
        context = read_knowledge_base(text)
        model = select_model(task_type, text)
        system_prompt = get_system_prompt(task_type, context=context)

        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": text}]

        response = litellm.completion(
            model=model,
            messages=messages
        )

        result = response.choices[0].message.content
        add_message_to_history(session_id, "user", text)
        add_message_to_history(session_id, "assistant", result)

        update_job(job_id, "completed", {"response": result, "model_used": model})

    except Exception as e:
        update_job(job_id, "failed", str(e))
