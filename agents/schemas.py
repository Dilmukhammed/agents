from pydantic import BaseModel, Field
from typing import List, Union

class Step(BaseModel):
    step_id: str = Field(..., description="A unique identifier for the step, e.g., '1.1'.")
    task_description: str = Field(..., description="A very specific action for the agent to perform.")
    assigned_agent: str = Field(..., description="The name of the agent assigned to this task.")
    expected_output: str = Field(..., description="A clear description of the data/artifact this step will produce.")
    dependencies: List[str] = Field(..., description="A list of step_ids that this step depends on.")

class Phase(BaseModel):
    phase_name: str = Field(..., description="The name of the phase, e.g., 'Research and Content Strategy'.")
    steps: List[Step]

class SuccessPlan(BaseModel):
    status: str = "success"
    plan_title: str = Field(..., description="An optimized title for the user's goal.")
    phases: List[Phase]

class FailurePlan(BaseModel):
    status: str = "failure"
    reason: str = Field(..., description="A clear, concise explanation of why the plan cannot be created.")
    missing_capabilities: List[str] = Field(..., description="A list of required but unavailable capabilities.")
    questions_for_user: List[str] = Field(..., description="Specific questions to the user for clarification.")

# A type that can be either a success or a failure plan
Plan = Union[SuccessPlan, FailurePlan]
