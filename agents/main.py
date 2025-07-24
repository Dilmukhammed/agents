import asyncio
import os
import json
from typing import List, Dict, Union
from openai import AsyncOpenAI

# A real LLM class using openai SDK
class LLM:
    def __init__(self, model_config: Dict, provider_config: Dict, system_prompt: str = ""):
        self.model_name = model_config['model_name']
        self.provider_name = model_config['provider']
        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(
            api_key=provider_config['api_key'],
            base_url=provider_config.get('base_url')
        )

    async def _generate(self, user_prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error from {self.provider_name} ({self.model_name}): {e}"

    async def generate_plan(self, user_request: str) -> str:
        prompt = f"Пожалуйста, составь план для следующего запроса: {user_request}"
        return await self._generate(prompt)

    async def analyze_and_critique(self, all_plans: str) -> str:
        prompt = f"""Вот все планы, предложенные другими моделями:\n{all_plans}\n
Проанализируй их, найди недостатки и сильные стороны. Защити свой план и предложи улучшения для других.
Твоя задача - конструктивная критика и совместная работа над созданием лучшего плана."""
        return await self._generate(prompt)

    async def generate_final_plan(self, all_plans_and_critiques: str) -> str:
        prompt = f"""На основе всех предложенных планов и последующего обсуждения, создай единый, финальный и наилучший план.
Вот вся история обсуждения:\n{all_plans_and_critiques}"""
        return await self._generate(prompt)


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
        for i, plan in enumerate(plans):
            f.write(f"--- План модели {models[i].provider_name}/{models[i].model_name} ---\n{plan}\n\n")

async def stage_2_discussion_and_refinement(models: List[LLM]):
    """Stage 2: Models discuss and refine plans."""
    with open("all_plans.txt", "r+", encoding="utf-8") as f:
        all_plans = f.read()
        for model in models:
            critique = await model.analyze_and_critique(all_plans)
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
    s1_models = []
    for i, model_conf in enumerate(config['stage_1_models']):
        provider_conf = providers[model_conf['provider']]
        prompt = stage_1_prompts[i] if isinstance(stage_1_prompts, list) and i < len(stage_1_prompts) else stage_1_prompts
        s1_models.append(LLM(model_config=model_conf, provider_config=provider_conf, system_prompt=prompt))

    s2_models = []
    for i, model_conf in enumerate(config['stage_2_models']):
        provider_conf = providers[model_conf['provider']]
        prompt = stage_2_prompts[i] if isinstance(stage_2_prompts, list) and i < len(stage_2_prompts) else stage_2_prompts
        s2_models.append(LLM(model_config=model_conf, provider_config=provider_conf, system_prompt=prompt))

    s3_model_conf = config['stage_3_model']
    s3_provider_conf = providers[s3_model_conf['provider']]
    final_model = LLM(model_config=s3_model_conf, provider_config=s3_provider_conf, system_prompt=stage_3_prompt)


    # --- Run Stages ---
    print("--- Этап 1: Генерация первоначальных планов ---")
    await stage_1_generate_initial_plans(user_request, s1_models)
    print("Планирование завершено. Результаты в all_plans.txt")

    print("\n--- Этап 2: Обсуждение и доработка ---")
    await stage_2_discussion_and_refinement(s2_models)
    print("Обсуждение завершено. Результаты в all_plans.txt")

    print("\n--- Этап 3: Создание финального плана ---")
    await stage_3_generate_final_plan(final_model)
    print("Финальный план создан. Результат в final_plan.txt")


if __name__ == "__main__":
    # Note: For this to run, you need to replace placeholder API keys in config.json
    # with your actual keys.
    # The current implementation will likely fail due to invalid API keys.
    # This is a framework for you to build upon.
    print("Запуск основного скрипта. Убедитесь, что вы указали правильные API ключи в config.json")
    # asyncio.run(main())
    print("Выполнение закомментировано, чтобы избежать ошибок с неверными API ключами.")
    print("Раскомментируйте asyncio.run(main()) и настройте config.json для реального использования.")
