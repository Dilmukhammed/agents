from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Step(BaseModel):
    step_id: str = Field(..., description="A unique identifier for the step, e.g., '1.1'.")
    task_description: str = Field(..., description="A very specific action for the agent to perform.")
    assigned_agent: str = Field(..., description="The name of the agent assigned to this task.")
    expected_output: str = Field(..., description="A clear description of the data/artifact this step will produce.")
    dependencies: List[str] = Field(..., description="A list of step_ids that this step depends on.")

class Phase(BaseModel):
    phase_name: str = Field(..., description="The name of the phase, e.g., 'Research and Content Strategy'.")
    steps: List[Step]

class PlanResponse(BaseModel):
    status: Literal["success", "failure"] = Field(..., description="Indicates whether the plan creation was successful.")

    # Fields for a successful plan
    plan_title: Optional[str] = Field(None, description="An optimized title for the user's goal.")
    phases: Optional[List[Phase]] = Field(None, description="The list of phases in the plan.")

    # Fields for a failed plan
    reason: Optional[str] = Field(None, description="A clear, concise explanation of why the plan cannot be created.")
    missing_capabilities: Optional[List[str]] = Field(None, description="A list of required but unavailable capabilities.")
    questions_for_user: Optional[List[str]] = Field(None, description="Specific questions to the user for clarification.")
