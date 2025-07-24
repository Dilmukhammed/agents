import asyncio
import os
import json
from typing import List, Dict, Union, Type
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .schemas import PlanResponse, DebateResponse, FinalPlan

# A real LLM class using openai SDK
class LLM:
    def __init__(self, model_config: Dict, provider_config: Dict, system_prompt: str = ""):
        self.model_id = f"{model_config['provider']}/{model_config['model_name']}"
        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(
            api_key=provider_config['api_key'],
            base_url=provider_config.get('base_url')
        )
        self.model_params = {
            "model": model_config['model_name'],
            "temperature": model_config.get('temperature'),
            "top_p": model_config.get('top_p'),
            "max_tokens": model_config.get('max_tokens', 4096), # Default max_tokens
        }
        self.model_params = {k: v for k, v in self.model_params.items() if v is not None}

    async def _generate_and_parse_json(self, user_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        prompt_with_json_instructions = f"""{user_prompt}

Your response MUST be a single JSON object that conforms to the following Pydantic schema:
```json
{response_model.model_json_schema()}
```
"""
        try:
            response = await self.client.chat.completions.create(
                **self.model_params,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt_with_json_instructions},
                ],
                response_format={"type": "json_object"},
            )

            json_text = response.choices[0].message.content
            return response_model.model_validate_json(json_text)

        except (json.JSONDecodeError, ValidationError) as e:
            if response_model == PlanResponse:
                return PlanResponse(status="failure", reason=f"JSON parsing/validation error: {e}", missing_capabilities=[], questions_for_user=["The model returned an invalid JSON structure. Please try again."])
            elif response_model == DebateResponse:
                return DebateResponse(move="debate", commentary=None, final_plan=None)
            else:
                raise e
        except Exception as e:
            if response_model == PlanResponse:
                return PlanResponse(status="failure", reason=f"API Error: {e}", missing_capabilities=[], questions_for_user=[])
            elif response_model == DebateResponse:
                return DebateResponse(move="debate", commentary=None, final_plan=None)
            else:
                raise e

    async def generate_plan(self, user_request: str) -> PlanResponse:
        prompt = f"Generate a plan for this request: {user_request}. Available agents: AgentNameFromList, AnotherAgentFromList."
        return await self._generate_and_parse_json(prompt, response_model=PlanResponse)

    async def debate(self, history: str) -> DebateResponse:
        prompt = f"""You are in a debate with other AI models. Your goal is to collaboratively produce the best possible plan.
Your ID is: {self.model_id}.
The debate history so far:
{history}

Review the history, attack weak plans, defend your own, and concede good points.
If you believe your plan is now the best, you can submit it as final.
"""
        return await self._generate_and_parse_json(prompt, response_model=DebateResponse)

    async def generate_final_plan(self, submitted_plans: List[FinalPlan]) -> PlanResponse:
        submitted_plans_json = [plan.model_dump() for plan in submitted_plans]
        prompt = f"""You are the final judge. Multiple AI models have submitted their final plans after a debate.
Your task is to synthesize the best ideas from all of them and create one, single, consolidated master plan.
The submitted plans are in the following JSON array:
{json.dumps(submitted_plans_json, indent=2)}

Your final output must be a single, coherent, and actionable plan in the standard PlanResponse JSON format.
"""
        return await self._generate_and_parse_json(prompt, response_model=PlanResponse)


def load_system_prompts(file_path: str) -> Union[str, List[str]]:
    if not os.path.exists(file_path): return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = f.read().strip().split('\n---\n')
    return prompts if len(prompts) > 1 else prompts[0]

def load_config(file_path: str = "config.json") -> Dict:
    with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)

async def stage_1_generate_initial_plans(user_request: str, models: List[LLM]):
    tasks = [model.generate_plan(user_request) for model in models]
    plans = await asyncio.gather(*tasks)

    initial_discussion = ""
    with open("all_plans.txt", "w", encoding="utf-8") as f:
        f.write("--- Stage 1: Initial Plans ---\n\n")
        for i, plan in enumerate(plans):
            model_id = models[i].model_id
            plan_json = plan.model_dump_json(indent=2)
            f.write(f"--- Plan from {model_id} ---\n{plan_json}\n\n")
            initial_discussion += f"--- Plan from {model_id} ---\n{plan_json}\n\n"

    return initial_discussion

async def stage_2_debate(models: List[LLM], initial_discussion: str):
    active_models = list(models)
    submitted_final_plans: List[FinalPlan] = []
    debate_history = initial_discussion
    round_num = 1

    with open("all_plans.txt", "a", encoding="utf-8") as f:
        while len(active_models) > 1:
            round_header = f"\n--- Stage 2: Debate Round {round_num} ---\n\n"
            f.write(round_header)
            print(f"--- Stage 2: Debate Round {round_num} | Active Models: {len(active_models)} ---")

            tasks = [model.debate(debate_history) for model in active_models]
            responses = await asyncio.gather(*tasks)

            current_round_moves = ""
            models_to_remove = []

            for i, response in enumerate(responses):
                model = active_models[i]
                response_json = response.model_dump_json(indent=2)
                move_text = f"--- Move from {model.model_id} ---\n{response_json}\n\n"
                f.write(move_text)
                current_round_moves += move_text

                if response.move == "submit_final_plan" and response.final_plan:
                    submitted_final_plans.append(response.final_plan)
                    models_to_remove.append(model)
                    print(f"Model {model.model_id} has submitted a final plan and exited the debate.")

            debate_history += current_round_moves
            active_models = [m for m in active_models if m not in models_to_remove]
            round_num += 1

            if not any(res.move == "debate" for res in responses):
                print("Debate stalled. No new 'debate' moves. Ending debate.")
                break

    print(f"Debate finished. {len(submitted_final_plans)} final plans were submitted.")
    return submitted_final_plans

async def stage_3_generate_final_plan(final_model: LLM, submitted_plans: List[FinalPlan]):
    if not submitted_plans:
        print("No final plans were submitted. Skipping final consolidation.")
        with open("final_plan.txt", "w", encoding="utf-8") as f:
            f.write(json.dumps({"status": "failure", "reason": "No final plans were submitted during the debate."}, indent=2))
        return

    with open("all_plans.txt", "a", encoding="utf-8") as f:
        f.write("\n--- Stage 3: Submitted Final Plans for Consolidation ---\n\n")
        for i, plan in enumerate(submitted_plans):
            plan_text = plan.model_dump_json(indent=2)
            f.write(f"--- Submitted Plan {i+1} ---\n{plan_text}\n\n")

    final_plan = await final_model.generate_final_plan(submitted_plans)
    with open("final_plan.txt", "w", encoding="utf-8") as f:
        f.write(final_plan.model_dump_json(indent=2))

async def main():
    config = load_config()
    providers = config['providers']
    user_request = "Составь план для завоевания мира."

    s1_prompts = load_system_prompts('system_prompts/stage_1_prompts.txt')
    s2_prompts = load_system_prompts('system_prompts/stage_2_prompts.txt')
    s3_prompt = load_system_prompts('system_prompts/stage_3_prompts.txt')

    s1_models = [LLM(c, providers[c['provider']], s1_prompts[i % len(s1_prompts)] if isinstance(s1_prompts, list) else s1_prompts) for i, c in enumerate(config['stage_1_models'])]
    s2_models = [LLM(c, providers[c['provider']], s2_prompts[i % len(s2_prompts)] if isinstance(s2_prompts, list) else s2_prompts) for i, c in enumerate(config['stage_2_models'])]
    final_model = LLM(config['stage_3_model'], providers[config['stage_3_model']['provider']], s3_prompt)

    print("--- Этап 1: Генерация первоначальных планов ---")
    initial_discussion = await stage_1_generate_initial_plans(user_request, s1_models)
    print("Планирование завершено. Результаты в all_plans.txt")

    print("\n--- Этап 2: Дебаты ---")
    submitted_plans = await stage_2_debate(s2_models, initial_discussion)
    print("Дебаты завершены. Результаты в all_plans.txt")

    print("\n--- Этап 3: Создание финального плана ---")
    await stage_3_generate_final_plan(final_model, submitted_plans)
    print("Финальный план создан. Результат в final_plan.txt")

if __name__ == "__main__":
    print("Запуск основного скрипта. Убедитесь, что вы указали правильные API ключи в config.json")
    # asyncio.run(main())
    print("Выполнение закомментировано. Раскомментируйте для реального использования.")
