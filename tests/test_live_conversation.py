"""Optional provider evaluation; never silently uses a mock or a rule fallback."""
import os
import pytest
from app.conversation import ConversationEngine
from app.interpreter import RuleInterpreter
from app.repository import Repository
from app.workflow import ConversationService

pytestmark = pytest.mark.skipif(os.getenv('RUN_LIVE_EVAL') != '1', reason='Opt-in live model evaluation')


def test_live_compound_question_and_implied_emotion():
    service = ConversationService(Repository(), RuleInterpreter())
    service.respond('live', 'Margaret Chen, 1985-03-15, 4472')
    service.conversation = ConversationEngine()
    reply, state = service.respond('live', 'Why is my appeal denied when did this happen why was I not notified for it?')
    print('\nLIVE REPLY:', reply)
    assert state.response_mode == 'model', 'Provider failed or draft was rejected; fallback is not a live-eval pass'
    assert state.last_emotion != 'neutral'
    assert 'pathology' in reply.lower()
    assert 'no appeal decision' in reply.lower()
    assert 'no denial-decision date' in reply.lower()
    assert 'notif' in reply.lower() or 'notice' in reply.lower()
    assert not service.outbox
    reply, state = service.respond('live', 'I have spent the whole morning on this and nobody is listening. I cannot get those papers from the clinic. What am I supposed to do now?')
    print('\nLIVE CONTEXTUAL FOLLOWUP:', reply)
    assert state.response_mode == 'model'
    assert state.last_emotion != 'neutral'
    alternatives = service._evidence(state)['document_alternatives'].values()
    assert any(value in reply for value in alternatives), 'Must address the new document-access obstacle'
    assert not service.outbox
    # Schema/reviewer success is not proof of factual correctness: inspect printed reply too.


def test_live_unseen_emotion_without_keywords():
    service = ConversationService(Repository(), RuleInterpreter(), conversation=ConversationEngine())
    reply, state = service.respond('live', 'I have spent the whole morning on this and nobody is listening to me.')
    print('\nLIVE PREVERIFICATION REPLY:', reply)
    assert state.response_mode == 'model'
    assert state.last_emotion != 'neutral'
    assert not state.verified
    assert 'pathology' not in reply.lower()
    assert state.refusals == 0
