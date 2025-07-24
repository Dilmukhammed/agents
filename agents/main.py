import asyncio
import os
from typing import List, Dict, Union

# Placeholder for a real LLM API
class MockLLM:
    def __init__(self, model_id: str, system_prompt: str = ""):
        self.model_id = model_id
        self.system_prompt = system_prompt

    async def generate_plan(self, user_request: str) -> str:
        # Simulate API call
        await asyncio.sleep(1)
        return f"План от модели {self.model_id}:\n1. Сделать А.\n2. Сделать Б.\n"

    async def analyze_and_critique(self, all_plans: str) -> str:
        # Simulate API call
        await asyncio.sleep(1)
        return f"Критика от {self.model_id}:\nПлан от модели 2 слишком сложный. Мой план лучше."

    async def generate_final_plan(self, all_plans: str) -> str:
        # Simulate API call
        await asyncio.sleep(1)
        return f"Финальный план:\n1. Объединить лучшие части планов.\n2. Выполнить объединенный план."

def load_system_prompts(file_path: str) -> Union[str, List[str]]:
    """Loads system prompts from a file."""
    if not os.path.exists(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = f.read().strip().split('\n---\n')
    return prompts if len(prompts) > 1 else prompts[0]

async def stage_1_generate_initial_plans(user_request: str, models: List[MockLLM]):
    """Stage 1: Generate initial plans from all models."""
    tasks = [model.generate_plan(user_request) for model in models]
    plans = await asyncio.gather(*tasks)
    with open("all_plans.txt", "w", encoding="utf-8") as f:
        for i, plan in enumerate(plans):
            f.write(f"--- План модели {i+1} ---\n{plan}\n\n")

async def stage_2_discussion_and_refinement(models: List[MockLLM]):
    """Stage 2: Models discuss and refine plans."""
    with open("all_plans.txt", "r+", encoding="utf-8") as f:
        all_plans = f.read()
        for model in models:
            critique = await model.analyze_and_critique(all_plans)
            f.write(f"--- Комментарий от модели {model.model_id} ---\n{critique}\n\n")

async def stage_3_generate_final_plan(final_model: MockLLM):
    """Stage 3: Generate the final, consolidated plan."""
    with open("all_plans.txt", "r", encoding="utf-8") as f:
        all_plans_and_critiques = f.read()
    final_plan = await final_model.generate_final_plan(all_plans_and_critiques)
    with open("final_plan.txt", "w", encoding="utf-8") as f:
        f.write(final_plan)

async def main():
    # --- Configuration ---
    num_models = 5
    user_request = "Составь план для завоевания мира."

    # --- Load System Prompts ---
    stage_1_prompts = load_system_prompts('system_prompts/stage_1_prompts.txt')
    stage_2_prompts = load_system_prompts('system_prompts/stage_2_prompts.txt')
    stage_3_prompt = load_system_prompts('system_prompts/stage_3_prompts.txt')

    # --- Initialize Models ---
    models = []
    for i in range(num_models):
        # Flexible system prompt assignment
        s1_prompt = stage_1_prompts[i] if isinstance(stage_1_prompts, list) and i < len(stage_1_prompts) else stage_1_prompts
        models.append(MockLLM(model_id=str(i+1), system_prompt=s1_prompt))

    # --- Stage 1 ---
    print("--- Этап 1: Генерация первоначальных планов ---")
    await stage_1_generate_initial_plans(user_request, models)
    print("Планирование завершено. Результаты в all_plans.txt")

    # --- Stage 2 ---
    print("\n--- Этап 2: Обсуждение и доработка ---")
    for i, model in enumerate(models):
         s2_prompt = stage_2_prompts[i] if isinstance(stage_2_prompts, list) and i < len(stage_2_prompts) else stage_2_prompts
         model.system_prompt = s2_prompt
    await stage_2_discussion_and_refinement(models)
    print("Обсуждение завершено. Результаты в all_plans.txt")


    # --- Stage 3 ---
    print("\n--- Этап 3: Создание финального плана ---")
    final_model = MockLLM(model_id="final", system_prompt=stage_3_prompt)
    await stage_3_generate_final_plan(final_model)
    print("Финальный план создан. Результат в final_plan.txt")

if __name__ == "__main__":
    asyncio.run(main())
