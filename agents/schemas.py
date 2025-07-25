from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# --- Stage 1: Initial Plan Schemas ---

class ToolCall(BaseModel):
    mcp_server: str = Field(..., description="The MCP server to route the tool call to.")
    tool_name: str = Field(..., description="The name of the tool to be called.")
    parameters: dict = Field(..., description="The parameters for the tool call.")

class Task(BaseModel):
    step_id: str = Field(..., description="A unique identifier for the step, e.g., '1.1'.")
    task_description: str = Field(..., description="A very specific action for the agent to perform.")
    assigned_agent: str = Field(..., description="The name of the agent assigned to this task.")
    tool_call: ToolCall = Field(..., description="The tool call to be executed for this task.")
    expected_output: str = Field(..., description="A clear description of the data/artifact this step will produce.")
    dependencies: List[str] = Field(..., description="A list of step_ids that this step depends on.")

class Phase(BaseModel):
    phase_name: str = Field(..., description="The name of the phase, e.g., 'Research and Content Strategy'.")
    steps: List[Task]

class PlanResponse(BaseModel):
    status: Literal["success", "failure"] = Field(..., description="Indicates whether the plan creation was successful.")
    plan_title: Optional[str] = Field(None, description="An optimized title for the user's goal.")
    phases: Optional[List[Phase]] = Field(None, description="The list of phases in the plan.")
    reason: Optional[str] = Field(None, description="A clear, concise explanation of why the plan cannot be created.")
    missing_capabilities: Optional[List[str]] = Field(None, description="A list of required but unavailable capabilities.")
    questions_for_user: Optional[List[str]] = Field(None, description="Specific questions to the user for clarification.")

# --- Stage 2: Debate Schemas ---

class Attack(BaseModel):
    target_model_id: str = Field(..., description="The ID of the model being critiqued.")
    argument: str = Field(..., description="The specific critique of the target model's plan.")

class Defense(BaseModel):
    responding_to_critique_from: str = Field(..., description="The ID of the model whose critique is being addressed.")
    argument: str = Field(..., description="The defense of one's own plan.")

class Concession(BaseModel):
    conceding_to: str = Field(..., description="The ID of the model to which a point is being conceded.")
    point: str = Field(..., description="The specific point being conceded.")

class Commentary(BaseModel):
    attacks: Optional[List[Attack]] = Field(None, description="Critiques of other models' plans.")
    defenses: Optional[List[Defense]] = Field(None, description="Defenses against critiques.")
    concessions: Optional[List[Concession]] = Field(None, description="Points conceded to other models.")

class FinalPlan(BaseModel):
    plan_title: str = Field(..., description="The final, revised title for the user's goal.")
    phases: List[Phase]

class DebateResponse(BaseModel):
    move: Literal["debate", "submit_final_plan"] = Field(..., description="The type of move being made.")
    commentary: Optional[Commentary] = Field(None, description="The commentary for the debate move.")
    final_plan: Optional[FinalPlan] = Field(None, description="The final submitted plan.")
