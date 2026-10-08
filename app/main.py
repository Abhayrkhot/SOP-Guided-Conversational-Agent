import hmac
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from .interpreter import RuleInterpreter
from .conversation import ConversationEngine
from .repository import Repository
from .workflow import ConversationService, ConcurrentTurnError
from .store import Store

app = FastAPI(title='Insurance SOP Agent')
service = ConversationService(Repository(), RuleInterpreter(), Store(os.getenv('SESSION_DB', ':memory:')), conversation=ConversationEngine())

class Message(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    text: str = Field(min_length=1, max_length=4000)

@app.post('/api/chat')
@app.post('/api/chat-v2')
def chat(message: Message):
    if not message.text.strip():
        raise HTTPException(422, 'Message cannot be blank')
    if message.session_id and not service.exists(message.session_id):
        raise HTTPException(404, 'Session expired or not found. Start a new chat.')
    sid = message.session_id or str(uuid.uuid4())
    try:
        reply, state = service.respond(sid, message.text)
    except ConcurrentTurnError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {'session_id': sid, 'reply': reply, 'phase': state.phase,
            'verified': state.verified, 'captured_fields': list(state.identity),
            'remembered_hint': state.hint.model_dump(exclude_none=True),
            'case_id': state.case_id, 'email_choice': state.email_choice, 'handoff': state.handoff, 'response_mode': state.response_mode}

@app.get('/api/health')
def health():
    return {'ok': True, 'mode': service.conversation.mode}

@app.get('/api/outbox')
def outbox(authorization: str | None = Header(default=None)):
    token = os.getenv('ADMIN_API_TOKEN')
    if not token:
        raise HTTPException(404, 'Not found')
    if not authorization or not hmac.compare_digest(authorization, 'Bearer ' + token):
        raise HTTPException(401, 'Authentication required')
    return {'count': len(service.outbox), 'note': 'Demo only; no email delivery', 'messages': service.outbox}

@app.get('/')
def ui():
    return HTMLResponse((Path(__file__).parent / 'index.html').read_text())
