from datetime import date
import re
from .models import Phase, Session, Hint
from .interpreter import RuleInterpreter, BUSINESS_INTENTS, consent_of, decision_topics
from .store import Store
from .conversation import redact

TRANSITIONS = {Phase.VERIFY_ID: Phase.RESOLVE_INTENT, Phase.RESOLVE_INTENT: Phase.PROCESS_CASE,
               Phase.PROCESS_CASE: Phase.POST_PROCESS, Phase.POST_PROCESS: Phase.COMPLETE}

def advance(state, target):
    if TRANSITIONS.get(state.phase) != target:
        raise ValueError(f'Illegal transition: {state.phase} -> {target}')
    if not (state.verified and state.party_id):
        raise PermissionError('Verified customer required')
    if target in (Phase.PROCESS_CASE, Phase.POST_PROCESS, Phase.COMPLETE) and not state.case_id:
        raise ValueError('Claim selection required')
    if target == Phase.COMPLETE and state.email_choice is None:
        raise ValueError('Email decision required')
    state.phase = target

def claim_summary(claim):
    summary = f"Claim {claim['case_id']} ({claim['case_type']}) was filed on {claim['created_at']} and is {claim['status']}."
    if claim.get('denial_reason'):
        summary += f" The denial reason is: {claim['denial_reason']}."
    if claim.get('appeal_deadline'):
        summary += f" The recorded appeal deadline is {claim['appeal_deadline']}."
    return summary

LABELS = {'name': 'full name', 'dob': 'date of birth', 'phone': 'phone number', 'email': 'email address', 'id_last4': 'last four ID/SSN digits'}
OFFER = ' Would you like a human representative? Reply yes or no. This demo can record the request but cannot transfer a live call.'

class ConcurrentTurnError(RuntimeError):
    pass


class ConversationService:
    def __init__(self, repo, interpreter, store=None, today=date.today, conversation=None):
        self.repo, self.interpreter = repo, interpreter
        self.store = store or Store()
        self.today = today
        self.rules = RuleInterpreter()
        self.conversation = conversation

    @property
    def outbox(self):
        return self.store.messages()

    def exists(self, sid):
        with self.store.lock:
            return self.store.get(sid) is not None

    def respond(self, session_id, text):
        with self.store.lock:
            state = self.store.get(session_id) or Session()
        revision = state.revision
        active = state.phase != Phase.COMPLETE and state.handoff != 'requested'
        identity = {**state.identity, **self.rules.interpret(text, state).identity}
        safe_text = redact(text, identity)
        plan = None
        state.response_mode = 'rules'
        if self.conversation and self.conversation.mode == 'model' and active:
            state.response_mode = 'fallback'
            try:
                plan = self.conversation.understand(safe_text, state)
                # Emotion needs evidence from this turn, not model speculation.
                if plan.emotion != 'neutral' and (not plan.emotion_evidence or plan.emotion_evidence.casefold() not in safe_text.casefold()):
                    plan.emotion, plan.intensity = 'neutral', 0
                state.last_emotion = plan.emotion
                state.distress_turns = state.distress_turns + 1 if plan.intensity >= 2 else 0
                if plan.questions and plan.scope != 'unrelated':
                    state.pending_questions = [redact(q, identity) for q in plan.questions]
                state.response_mode = 'model'
            except Exception:
                self.conversation.failed()
        previous_discussion = list(state.discussed)
        previous_next_steps = list(state.next_steps)
        reply = self._respond(session_id, state, text, plan)
        if (plan and state.verified and state.case_id
                and state.phase in (Phase.PROCESS_CASE, Phase.POST_PROCESS)
                and state.handoff != 'requested' and plan.scope != 'unrelated'
                and (plan.questions or state.pending_questions)):
            required = ''
            if state.phase == Phase.POST_PROCESS:
                required = 'Would you like the summary sent to your registered email address? Reply send or skip. Delivery is simulated.'
            elif state.finish_offered:
                required = 'Are you ready to finish the claim discussion? Reply yes or no.'
            if state.handoff == 'offered':
                required = 'Would you like a human representative? Reply yes or no. This demo cannot transfer live calls.'
            try:
                composed = self.conversation.compose(safe_text, state, plan, self._evidence(state), required)
                reply = composed.text
                state.discussed = previous_discussion
                state.next_steps = previous_next_steps
                if composed.suggested_next_step and composed.suggested_next_step not in state.next_steps:
                    state.next_steps.append(composed.suggested_next_step)
                if composed.offers_human:
                    state.handoff = 'offered'
                if reply not in state.discussed:
                    state.discussed.append(reply)
                state.pending_questions = []
            except Exception:
                self.conversation.failed()
                state.response_mode = 'fallback'
        if active:
            state.history.extend([{'role':'user', 'content':safe_text}, {'role':'assistant', 'content':reply}])
            state.history = state.history[-12:]
            state.discussed = state.discussed[-20:]
        with self.store.turn():
            current = self.store.get(session_id)
            if (current.revision if current else 0) != revision:
                if current and current.phase == Phase.COMPLETE:
                    return 'This conversation is complete. Please start a new session.', current
                raise ConcurrentTurnError('The conversation changed while this response was prepared. Please retry your message.')
            if state.pending_email_body is not None:
                self.store.queue(session_id, self.repo.person(state.party_id)['email'], state.pending_email_body)
                state.pending_email_body = None
            state.revision += 1
            self.store.save(session_id, state)
        return reply, state


    def _evidence(self, state):
        if not state.verified or not state.party_id or not state.case_id:
            raise PermissionError('Verified owned claim required for model evidence')
        claim = self.repo.claim(state.party_id, state.case_id)
        if not claim:
            raise PermissionError('Selected claim not owned by verified customer')
        guidance = self.repo.guidance
        docs = claim.get('documents_needed', [])
        aliases = {'pathology report':'original pathology report', 'office note':'treating provider office note'}
        evidence = {
            'claim': claim,
            'record_limits': 'Absent claim fields are unknown. Filing date is not decision date. Claim denial is not appeal denial. No notification history is recorded.',
            'today': self.today().isoformat(),
            'workflow': {'phase':state.phase.value, 'email_choice':state.email_choice,
                         'handoff':state.handoff, 'delivery':'demo only, no real delivery or live transfer'},
        }
        if docs:
            evidence['submission_guidance'] = guidance['default_guidance']['en']
            evidence['document_alternatives'] = {
                doc:guidance['document_alternative_guidance'].get(aliases.get(doc, doc), guidance['document_alternative_guidance']['default'])['en']
                for doc in docs}
            for item in guidance['claim_followup_guidance']:
                evidence[item['topic']] = item['en'].format(case_id=claim['case_id'], documents=', '.join(docs),
                    average_processing_time_after_submission=guidance['claim_followup_settings']['average_processing_time_after_submission']['en'])
        statements = {
            'claim.status': f"Claim {claim['case_id']} is recorded as {claim['status']}.",
            'claim.filed_date': f"The claim was filed on {claim['created_at']}; this is not a denial-decision date.",
            'missing.appeal_decision': 'The available record contains no appeal decision, so I cannot confirm whether an appeal was denied.',
            'missing.decision_date': 'No denial-decision date is recorded. The filing date and appeal deadline do not establish when a decision was made.',
            'missing.notification_history': 'No notification history is available, so I cannot confirm whether, when, or how a notice was sent, or why you did not receive one.',
            'missing.other': 'The available record does not answer that question. A representative would need to check the relevant information.',
            'claim.documents': ('The listed missing documents are: ' + ', '.join(docs) + '.') if docs else 'No missing documents are listed for this claim.',
        }
        if claim.get('denial_reason'):
            statements['claim.denial_reason'] = 'The recorded claim-denial reason is: ' + claim['denial_reason'] + '.'
        if claim.get('appeal_deadline'):
            deadline = claim['appeal_deadline']
            statements['claim.appeal_deadline'] = f'The recorded appeal deadline is {deadline}.'
            if date.fromisoformat(deadline) < self.today():
                statements['claim.appeal_deadline'] += ' It has passed; I cannot confirm eligibility for a late appeal.'
        if claim.get('net_pay'):
            statements['claim.net_pay'] = 'The recorded net payment is USD ' + claim['net_pay'] + '.'
        if claim.get('expected_reimbursement_amount'):
            statements['claim.expected_reimbursement'] = 'The recorded expected reimbursement is USD ' + claim['expected_reimbursement_amount'] + '.'
        for key, value in evidence.items():
            if key not in ('claim', 'record_limits', 'today', 'workflow', 'document_alternatives'):
                statements['guidance.' + key] = value
        for doc, guidance_text in evidence.get('document_alternatives', {}).items():
            statements['alternative.' + doc] = guidance_text
        evidence['fact_dependencies'] = {'missing.appeal_decision': [key for key in ('claim.status', 'claim.denial_reason') if key in statements]}
        evidence['approved_statements'] = statements
        return evidence

    def _respond(self, sid, state, text, plan=None):
        if state.phase == Phase.COMPLETE:
            return 'This conversation is complete. Please start a new session.'
        if state.handoff == 'requested':
            return 'Your human-support request is recorded in this demo. Please contact the insurer using the number on your policy or member portal. No live transfer has occurred.'
        trusted = self.rules.interpret(text, state)
        message = self.interpreter.interpret(text, state)
        # Defense in depth: even an arbitrary interpreter cannot supply identity or consent.
        message.identity = trusted.identity
        message.human = trusted.human
        if plan:
            message.unrelated = plan.scope == 'unrelated'
            if plan.emotion != 'neutral':
                message.emotion = plan.emotion
            if plan.wants_human or plan.intensity == 3 or state.distress_turns >= 3:
                state.handoff = 'offered'
            if plan.hint.case_id and plan.hint.case_id.upper() not in text.upper():
                plan.hint.case_id = None
            if plan.hint.year and str(plan.hint.year) not in text:
                plan.hint.year = None
            if state.phase in (Phase.VERIFY_ID, Phase.RESOLVE_INTENT):
                for field, value in plan.hint.model_dump(exclude_none=True).items():
                    setattr(state.hint, field, value)
        if message.intent not in BUSINESS_INTENTS:
            message.intent = trusted.intent
        if state.phase == Phase.POST_PROCESS:
            message.intent = trusted.intent
        if state.phase == Phase.VERIFY_ID:
            self._capture_identity(state, text, trusted.identity)
        if state.phase in (Phase.VERIFY_ID, Phase.RESOLVE_INTENT):
            if trusted.hint.case_id:
                state.hint = Hint(case_id=trusted.hint.case_id)
            else:
                if re.search(r'\b(actually|correction|instead|start over|clear.*hint)\b', text, re.I):
                    state.hint = Hint()
                for field, value in trusted.hint.model_dump(exclude_none=True).items():
                    setattr(state.hint, field, value)
            if message.intent in BUSINESS_INTENTS:
                state.original_intent = message.intent
                state.original_decision_topics = decision_topics(text, state.current_intent)
        acknowledgments = {
            'angry': 'I hear how upsetting this has been. Let’s work through what I can check. ',
            'frustrated': 'I understand why you want a clearer explanation. ',
            'anxious': 'We can take this one step at a time and check what is known. ',
            'sad': 'I’m sorry you’re dealing with this. ',
            'confused': 'Let me help make the next step clearer. ',
            'overwhelmed': 'We can slow down and focus on one step at a time. ',
        }
        empathy = acknowledgments.get(message.emotion, 'I’m sorry this has been stressful. We can take this one step at a time. ') if message.emotion else ''
        if message.human or (state.handoff == 'offered' and consent_of(text) == 'send_email'):
            state.handoff = 'requested'
            state.identity.clear()
            return empathy + 'Your request for human support is recorded in this demo. Contact the insurer using the number on your policy or member portal. No live transfer has occurred.'
        if state.handoff == 'offered' and consent_of(text) == 'skip_email':
            state.handoff = None
            return 'Understood. We can continue here. ' + self._prompt(state)
        if message.unrelated:
            state.irrelevant += 1
            if state.irrelevant >= 3:
                state.handoff = 'offered'
                return 'I can only help with insurance claims and related support.' + OFFER
            return 'I can only help with insurance claims and related support. ' + self._prompt(state)
        state.irrelevant = 0
        if state.phase == Phase.VERIFY_ID:
            reply = empathy + self._verify(state, message)
            if state.handoff == 'offered' and OFFER not in reply:
                reply += OFFER
            return reply
        if state.phase == Phase.RESOLVE_INTENT:
            return empathy + self._resolve(state)
        if state.phase == Phase.PROCESS_CASE:
            confirmed_finish = state.finish_offered and consent_of(text) == 'send_email'
            if state.finish_offered and consent_of(text) == 'skip_email':
                state.finish_offered = False
                return empathy + 'Of course, we can continue. What would you like to know?'
            if trusted.intent == 'finish' or confirmed_finish:
                state.finish_offered = False
                advance(state, Phase.POST_PROCESS)
                return empathy + 'Would you like an email summary of our discussion, claim status, and next steps sent to your registered email address? Reply send or skip. Delivery is simulated.'
            if plan and plan.wants_to_finish:
                state.finish_offered = True
                return empathy + 'Are you ready to finish the claim discussion? Reply yes or no.'
            reply = empathy + self._process(state, message.intent, text)
            if state.handoff == 'offered' and OFFER not in reply:
                reply += OFFER
            return reply
        return empathy + self._post(sid, state, message, text)

    def _capture_identity(self, state, text, identity):
        if re.search(r'\b(start over|clear identity)\b', text, re.I):
            state.identity.clear()
        aliases = {'name': 'name', 'dob': 'dob|date of birth', 'phone': 'phone', 'email': 'email', 'id_last4': 'ssn|id|last four|last 4', 'policy_number': 'policy'}
        for field, pattern in aliases.items():
            if re.search(rf'\b(?:forget|remove|ignore) (?:my |the )?(?:{pattern})\b', text, re.I):
                state.identity.pop(field, None)
        state.identity.update(identity)

    def _verify(self, state, message):
        if message.refusal:
            state.refusals += 1
        if state.refusals >= 3 or state.verification_failures >= 5:
            state.handoff = 'offered'
            return 'I cannot disclose claim details without verification. You can pause here or choose different identity fields.' + OFFER
        person = self.repo.verify(state.identity)
        if person:
            state.party_id, state.verified = person['party_id'], True
            state.identity.clear()
            advance(state, Phase.RESOLVE_INTENT)
            return 'Thank you, your identity is verified. ' + self._resolve(state)
        captured = [key for key in LABELS if key in state.identity]
        if len(captured) >= 3:
            state.verification_failures += 1
            return 'Those details did not verify your identity. Please correct a value, or say "remove my SSN" (or another field) and provide an alternative. I cannot identify which value differs.'
        if set(captured) == {'dob', 'id_last4'}:
            return ('I have your date of birth and last four ID digits. To verify your identity, '
                    'please provide your full name, including your surname, or your phone number or email address. '
                    'I need three matching details before I can discuss your claim.')
        missing = ', '.join(value for key, value in LABELS.items() if key not in state.identity)
        return f'Your claim information is private, so I need three matching identity details. I have {len(captured)} of the five fields. Please provide {3-len(captured)} more: {missing}. A policy number does not count toward the three. You can choose any three or ask for a human.'

    def _resolve(self, state):
        matches = self.repo.claims_for(state.party_id, state.hint)
        if len(matches) == 1:
            claim = matches[0]
            state.case_id = claim['case_id']
            advance(state, Phase.PROCESS_CASE)
            summary = claim_summary(claim)
            state.discussed.append(summary)
            if state.original_intent in ('denial', 'status'):
                state.current_intent = state.original_intent
                return summary + ' Would you like help with documents, submission, or appeal options?'
            if state.original_intent:
                return summary + ' ' + self._process(state, state.original_intent, '')
            return summary + ' What would you like to know?'
        if not matches:
            return 'No claim matches those details in your account. Please provide a claim ID, or say "actually" followed by the corrected type or month to replace the earlier hint.'
        options = '; '.join(f"{c['case_id']}: {c['case_type']}, filed {c['created_at']}, {c['status']}" for c in matches)
        return 'Which claim do you mean? ' + options

    def _process(self, state, intent, raw):
        claim = self.repo.claim(state.party_id, state.case_id)
        if not claim:
            raise PermissionError('Selected claim not owned by verified customer')
        requested = re.search(r'\bCL-\d+\b', raw, re.I)
        if requested and requested.group().upper() != state.case_id:
            return 'This conversation is about ' + state.case_id + '. Please start a new chat to verify and select another claim.'
        docs = claim.get('documents_needed', [])
        next_step = None
        if intent == 'denial_details':
            topics = decision_topics(raw, state.current_intent) if raw else state.original_decision_topics
            reply, next_step = self._decision_details(claim, topics)
        elif intent == 'appeal':
            deadline = claim.get('appeal_deadline')
            if deadline and date.fromisoformat(deadline) < self.today():
                reply = f'The recorded appeal deadline was {deadline} and has passed. I cannot confirm eligibility for a late appeal.'
            else:
                reply = f'The recorded appeal deadline is {deadline}.' if deadline else 'There is no appeal deadline recorded.'
            next_step = 'Ask a claims representative to review appeal eligibility and filing requirements.'
        elif intent == 'denial':
            reply = 'The recorded denial reason is: ' + claim['denial_reason'] + '.' if claim.get('denial_reason') else 'There is no denial reason recorded for this claim.'
        elif intent == 'documents':
            reply = 'The missing documents are: ' + ', '.join(docs) + '.' if docs else 'No missing documents are listed.'
            if docs:
                next_step = 'Gather ' + ', '.join(docs) + '.'
        elif intent == 'status':
            reply = f"Claim {claim['case_id']} is currently recorded as {claim['status']}."
        elif intent == 'payment':
            reply = f"The recorded net payment is USD {claim.get('net_pay', 'not available')}. The expected reimbursement is USD {claim.get('expected_reimbursement_amount', 'not available')}."
        elif intent in ('submission', 'alternatives', 'timing', 'format', 'receipt'):
            reply = self._guidance(claim, intent, raw)
            if intent == 'alternatives':
                state.alternative_attempts += 1
                if state.alternative_attempts >= 2:
                    state.handoff = 'offered'
                    reply += OFFER
            if docs:
                next_step = ('Request replacement copies of ' if intent == 'alternatives' else 'Prepare ') + ', '.join(docs) + ' and confirm submission requirements through your member portal or support team.'
        elif intent == 'next_steps':
            next_step = 'Gather ' + ', '.join(docs) + ' and ask a representative to review the claim.' if docs else 'Contact a representative if you need further review of this claim.'
            reply = 'I cannot promise a change in the claim outcome.'
        else:
            return 'I can help with claim status, denial reasons, documents, submission guidance, payments, and appeals. Please clarify your question, or say done when finished.'
        if next_step:
            reply += ' ' + next_step
            if next_step not in state.next_steps:
                state.next_steps.append(next_step)
        state.current_intent = intent
        if reply not in state.discussed:
            state.discussed.append(reply)
        return reply

    def _decision_details(self, claim, topics):
        parts = []
        if 'appeal_decision' in topics:
            parts.append('The available record does not contain an appeal decision, so I cannot confirm that your appeal was denied.')
        if 'reason' in topics:
            if claim.get('denial_reason'):
                parts.append('The recorded claim-denial reason is: ' + claim['denial_reason'] + '.')
            else:
                parts.append(f"The claim is recorded as {claim['status']}; no claim-denial reason is recorded.")
        if 'decision_date' in topics:
            parts.append('No denial-decision date is recorded. The filing date and appeal deadline do not tell me when a denial decision was made.')
        if 'notification' in topics:
            parts.append('No notification history is available in this record. I cannot confirm whether, when, or how a notice was sent, or why you did not receive one.')
        if not parts:
            parts.append('Would you like to check the denial reason, decision date, or notification history?')
        if any(topic in topics for topic in ('appeal_decision', 'decision_date', 'notification')):
            next_step = 'Ask a claims representative to check the decision history and notice-delivery records and provide a copy of the decision letter.'
        else:
            next_step = 'You can review the missing documents or ask a representative about appeal options.'
        return ' '.join(parts), next_step

    def _guidance(self, claim, intent, raw):
        g = self.repo.guidance
        if not claim.get('documents_needed'):
            return 'No missing documents are listed for this claim. A representative can confirm whether anything else is needed.'
        if intent == 'submission':
            return g['default_guidance']['en']
        if intent == 'alternatives':
            aliases = {'pathology report': 'original pathology report', 'office note': 'treating provider office note'}
            texts = []
            for doc in claim['documents_needed']:
                item = g['document_alternative_guidance'].get(aliases.get(doc, doc), g['document_alternative_guidance']['default'])
                if item['en'] not in texts:
                    texts.append(item['en'])
            return ' '.join(texts)
        topic = {'timing': 'processing_time_after_submission', 'format': 'file_format_requirements', 'receipt': 'receipt_confirmation'}[intent]
        if intent == 'timing' and re.search(r'when.*submit|how soon.*(?:submit|send)', raw, re.I):
            topic = 'submission_timing'
        template = next(x['en'] for x in g['claim_followup_guidance'] if x['topic'] == topic)
        return template.format(case_id=claim['case_id'], documents=', '.join(claim['documents_needed']),
            average_processing_time_after_submission=g['claim_followup_settings']['average_processing_time_after_submission']['en'])

    def _post(self, sid, state, message, raw):
        if message.intent == 'skip_email':
            state.email_choice = 'skipped'
            advance(state, Phase.COMPLETE)
            return 'Understood. No email will be sent. Thank you for contacting claims support.'
        if message.intent == 'send_email':
            claim = self.repo.claim(state.party_id, state.case_id)
            if not claim:
                raise PermissionError('Selected claim not owned by verified customer')
            body = claim_summary(claim) + '\nDiscussion:\n' + '\n'.join(state.discussed)
            body += '\nNext steps:\n' + ('\n'.join(state.next_steps) or 'No additional next steps were agreed.')
            state.pending_email_body = body
            state.email_choice = 'queued_demo'
            advance(state, Phase.COMPLETE)
            return 'I queued the summary in the demo outbox. No real email was sent. Thank you for contacting claims support.'
        if re.search(r'\b(address|different|another|change)\b', raw, re.I):
            return 'I can only queue the summary to the registered email address. No email has been queued. Reply send to use that address, or skip.'
        if message.intent in BUSINESS_INTENTS:
            return self._process(state, message.intent, raw) + ' Would you like the summary? Reply send or skip.'
        return 'The summary is optional and includes claim information. Reply send to queue it to your registered email address, or skip. No email has been queued.'

    def _prompt(self, state):
        if state.phase == Phase.VERIFY_ID:
            return 'Please provide any three identity fields, or ask for a human.'
        if state.phase == Phase.POST_PROCESS:
            return 'Would you like the summary? Reply send or skip.'
        return 'What would you like to know about your claim?'
