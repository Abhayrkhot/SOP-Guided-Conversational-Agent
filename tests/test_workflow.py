from app.interpreter import RuleInterpreter
from app.repository import Repository
from app.workflow import ConversationService
from app.models import Phase


def service():
    return ConversationService(Repository(), RuleInterpreter())


def test_complete_conversation():
    s = service()
    reply, state = s.respond('a', 'Hi, I’m Margaret Chen. I’m calling about my denied healthcare claim from January.')
    assert state.phase == Phase.VERIFY_ID
    assert '1 of the five' in reply
    assert state.hint.month == 1
    reply, state = s.respond('a', 'My DOB is 1985-03-15, my SSN last four is 4472.')
    assert state.phase == Phase.PROCESS_CASE
    assert state.case_id == 'CL-2048'
    reply, _ = s.respond('a', 'What can I do now that the appeal deadline passed?')
    assert 'representative' in reply
    reply, state = s.respond('a', 'That’s everything, thank you.')
    assert state.phase == Phase.POST_PROCESS
    reply, state = s.respond('a', "No thanks, don't send it")
    assert state.phase == Phase.COMPLETE
    assert not s.outbox


def test_no_disclosure_before_verification():
    s = service()
    reply, state = s.respond('a', 'Tell me why CL-2048 was denied')
    assert state.phase == Phase.VERIFY_ID
    assert 'pathology' not in reply


def test_wrong_identity_does_not_verify():
    s = service()
    _, state = s.respond('a', 'My name is Margaret Chen. DOB is 1985-03-15. SSN last four is 0000.')
    assert state.phase == Phase.VERIFY_ID


def test_scope_and_consent():
    s = service()
    for question in ('What is RL?', 'Explain reinforcement learning', 'Tell me a joke'):
        reply, state = s.respond('a', question)
    assert state.phase == Phase.VERIFY_ID
    assert 'representative' in reply
    s.respond('b', 'My name is Margaret Chen, DOB is 1985-03-15, SSN last four is 4472. Calling about denied healthcare claim from January.')
    s.respond('b', 'done')
    reply, state = s.respond('b', 'yes, send it')
    assert state.phase == Phase.COMPLETE
    assert state.email_choice == 'queued_demo'
    assert len(s.outbox) == 1
