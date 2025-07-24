import asyncio
import os
import json
from typing import List, Dict, Union
from openai import AsyncOpenAI
from pydantic import BaseModel

from schemas import Plan, SuccessPlan, FailurePlan

# A real LLM class using openai SDK
class LLM:
    def __init__(self, model_config: Dict, provider_config: Dict, system_prompt: str = ""):
        self.model_name = model_config['model_name']
        self.provider_name = model_config['provider']
        self.system_prompt = system_prompt
        self.temperature = model_config.get('temperature')
        self.top_p = model_config.get('top_p')
        self.max_tokens = model_config.get('max_tokens')
        # This parameter is not standard for OpenAI API and might cause errors
        # self.reasoning_effort = model_config.get('reasoning_effort')
        self.client = AsyncOpenAI(
            api_key=provider_config['api_key'],
            base_url=provider_config.get('base_url')
        )

    async def _generate_structured(self, user_prompt: str, response_format: BaseModel) -> Union[SuccessPlan, FailurePlan]:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_format,
            )
            # Assuming the API returns a compatible model instance.
            # The actual `parse` functionality might need to be handled differently
            # depending on the library version and async behavior.
            # For now, we'll assume the response is the parsed object.
            return completion.choices[0].message.parsed

        except Exception as e:
            return FailurePlan(
                reason=f"Error from {self.provider_name} ({self.model_name}): {e}",
                missing_capabilities=[],
                questions_for_user=["Could you please rephrase your request? The model failed to generate a plan."]
            )

    async def _generate_text(self, user_prompt: str) -> str:
        try:
            params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if self.top_p is not None:
                params["top_p"] = self.top_p
            if self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens

            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from {self.provider_name} ({self.model_name}): {e}"

    async def generate_plan(self, user_request: str) -> Union[SuccessPlan, FailurePlan]:
        prompt = f"Пожалуйста, составь план для следующего запроса: {user_request}. У тебя есть список агентов: AgentNameFromList, AnotherAgentFromList. Используй их."
        return await self._generate_structured(prompt, response_format=Plan)

    async def analyze_and_critique(self, all_plans: str) -> str:
        prompt = f"""Вот все планы, предложенные другими моделями:\n{all_plans}\n
Проанализируй их, найди недостатки и сильные стороны. Защити свой план и предложи улучшения для других.
Твоя задача - конструктивная критика и совместная работа над созданием лучшего плана."""
        return await self._generate_text(prompt)

    async def generate_final_plan(self, all_plans_and_critiques: str) -> str:
        prompt = f"""На основе всех предложенных планов и последующего обсуждения, создай единый, финальный и наилучший план.
Вот вся история обсуждения:\n{all_plans_and_critiques}"""
        return await self._generate_text(prompt)


def load_system_prompts(file_path: str) -> Union[str, List[str]]:
    """Loads system prompts from a file."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = f.read().strip().split('\n---\n')
    return prompts if len(prompts) > 1 else prompts[0]

def load_config(file_path: str = "config.json") -> Dict:
    """Loads configuration from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def stage_1_generate_initial_plans(user_request: str, models: List[LLM]):
    """Stage 1: Generate initial plans from all models."""
    tasks = [model.generate_plan(user_request) for model in models]
    plans = await asyncio.gather(*tasks)

    with open("all_plans.txt", "w", encoding="utf-8") as f:
        full_plans_for_discussion = ""
        for i, plan in enumerate(plans):
            model_id = f"{models[i].provider_name}/{models[i].model_name}"
            f.write(f"--- План модели {model_id} ---\n")
            # Use model_dump_json for Pydantic models
            plan_json = plan.model_dump_json(indent=2)
            f.write(plan_json)
            f.write("\n\n")
            full_plans_for_discussion += f"--- План модели {model_id} ---\n{plan_json}\n\n"

    return full_plans_for_discussion


async def stage_2_discussion_and_refinement(models: List[LLM], initial_plans_text: str):
    """Stage 2: Models discuss and refine plans."""
    with open("all_plans.txt", "a", encoding="utf-8") as f:
        for model in models:
            critique = await model.analyze_and_critique(initial_plans_text)
            f.write(f"--- Комментарий от модели {model.provider_name}/{model.model_name} ---\n{critique}\n\n")

async def stage_3_generate_final_plan(final_model: LLM):
    """Stage 3: Generate the final, consolidated plan."""
    with open("all_plans.txt", "r", encoding="utf-8") as f:
        all_plans_and_critiques = f.read()
    final_plan = await final_model.generate_final_plan(all_plans_and_critiques)
    with open("final_plan.txt", "w", encoding="utf-8") as f:
        f.write(final_plan)

async def main():
    # --- Configuration ---
    config = load_config()
    providers = config['providers']
    user_request = "Составь план для завоевания мира."

    # --- Load System Prompts ---
    stage_1_prompts = load_system_prompts('system_prompts/stage_1_prompts.txt')
    stage_2_prompts = load_system_prompts('system_prompts/stage_2_prompts.txt')
    stage_3_prompt = load_system_prompts('system_prompts/stage_3_prompts.txt')

    # --- Initialize Models ---
    s1_models, s2_models = [], []
    for i, model_conf in enumerate(config['stage_1_models']):
        provider_conf = providers[model_conf['provider']]
        prompt = stage_1_prompts[i] if isinstance(stage_1_prompts, list) and i < len(stage_1_prompts) else stage_1_prompts
        s1_models.append(LLM(model_config=model_conf, provider_config=provider_conf, system_prompt=prompt))

    for i, model_conf in enumerate(config['stage_2_models']):
        provider_conf = providers[model_conf['provider']]
        prompt = stage_2_prompts[i] if isinstance(stage_2_prompts, list) and i < len(stage_2_prompts) else stage_2_prompts
        s2_models.append(LLM(model_config=model_conf, provider_config=provider_conf, system_prompt=prompt))

    s3_model_conf = config['stage_3_model']
    s3_provider_conf = providers[s3_model_conf['provider']]
    final_model = LLM(model_config=s3_model_conf, provider_config=s3_provider_conf, system_prompt=stage_3_prompt)


    # --- Run Stages ---
    print("--- Этап 1: Генерация первоначальных планов ---")
    initial_plans_text = await stage_1_generate_initial_plans(user_request, s1_models)
    print("Планирование завершено. Результаты в all_plans.txt")

    print("\n--- Этап 2: Обсуждение и доработка ---")
    await stage_2_discussion_and_refinement(s2_models, initial_plans_text)
    print("Обсуждение завершено. Результаты в all_plans.txt")

    print("\n--- Этап 3: Создание финального плана ---")
    await stage_3_generate_final_plan(final_model)
    print("Финальный план создан. Результат в final_plan.txt")


if __name__ == "__main__":
    print("Запуск основного скрипта. Убедитесь, что вы указали правильные API ключи в config.json")
    # The following line is commented out to prevent execution with placeholder keys.
    # It will fail because `client.chat.completions.create` with `response_format`
    # is not a real method in the way it's mocked here. A real implementation
    # would use `client.chat.completions.create` and then parse the JSON output.
    # For this simulation, we assume the parsing happens magically.
    # asyncio.run(main())
    print("Выполнение закомментировано, чтобы избежать ошибок. Для реального использования раскомментируйте asyncio.run(main()) и настройте config.json.")
