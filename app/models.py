from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field

class Phase(str, Enum):
    VERIFY_ID = 'VERIFY_ID'
    RESOLVE_INTENT = 'RESOLVE_INTENT'
    PROCESS_CASE = 'PROCESS_CASE'
    POST_PROCESS = 'POST_PROCESS'
    COMPLETE = 'COMPLETE'

class Hint(BaseModel):
    case_id: str | None = None
    case_type: Literal['healthcare', 'dental', 'auto'] | None = None
    month: int | None = Field(default=None, ge=1, le=12)
    year: int | None = Field(default=None, ge=1900, le=2200)
    status: Literal['denied', 'closed', 'open'] | None = None

class Parsed(BaseModel):
    identity: dict[str, str] = Field(default_factory=dict)
    hint: Hint = Field(default_factory=Hint)
    intent: str | None = None
    emotion: str | None = None
    human: bool = False
    unrelated: bool = False
    refusal: bool = False

class Session(BaseModel):
    phase: Phase = Phase.VERIFY_ID
    identity: dict[str, str] = Field(default_factory=dict)
    verified: bool = False
    party_id: str | None = None
    hint: Hint = Field(default_factory=Hint)
    case_id: str | None = None
    original_intent: str | None = None
    original_decision_topics: list[str] = Field(default_factory=list)
    current_intent: str | None = None
    refusals: int = 0
    irrelevant: int = 0
    verification_failures: int = 0
    alternative_attempts: int = 0
    email_choice: str | None = None
    discussed: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    handoff: Literal['offered', 'requested'] | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    pending_questions: list[str] = Field(default_factory=list)
    last_emotion: str = 'neutral'
    distress_turns: int = 0
    finish_offered: bool = False
    response_mode: Literal['rules', 'model', 'fallback'] = 'rules'
    revision: int = 0
    pending_email_body: str | None = Field(default=None, exclude=True)
