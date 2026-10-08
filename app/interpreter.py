"""Model suggestions never authorize identity, consent, or phase changes."""
import re
from datetime import datetime
from typing import Protocol
from .models import Parsed, Hint, Phase

MONTHS = ('january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december')
BUSINESS_INTENTS = {'appeal', 'denial', 'documents', 'submission', 'alternatives', 'timing', 'format', 'receipt', 'status', 'next_steps', 'payment', 'denial_details'}

class Interpreter(Protocol):
    def interpret(self, text: str, state=None) -> Parsed: ...

def consent_of(text):
    t = text.lower().replace('’', "'").strip(' .!')
    if re.fullmatch(r"(?:no(?: thanks| thank you)?|skip(?: (?:it|the email|the summary))?|decline|not now)", t) or re.search(r"\b(?:don't|do not|never) (?:send|email|mail)\b", t):
        return 'skip_email'
    # Full utterance matching intentionally leaves questions/qualifications unresolved.
    if re.fullmatch(r'(yes(?: please)?(?:,? send (?:it|the summary))?|send(?: (?:it|the summary|the email))?(?: please)?|please send(?: (?:it|the summary|the email))?|sure|okay|ok|please do)', t):
        return 'send_email'
    return None

def decision_topics(text, previous=None):
    """Collect independent questions instead of choosing one keyword winner."""
    t = text.lower().replace('’', "'")
    denial = bool(re.search(r"\b(denied|denial|rejected|refused|turned down|not approved|didn't (?:you )?approve|did not (?:you )?approve)\b", t))
    notification = bool(re.search(r"\b(notified|notification|notify|notice|wasn't (?:i |we )?told|was (?:i |we )?not told|was not told|weren't (?:we )?told|never (?:told|tell)|didn't tell|did not tell)\b", t))
    date_question = bool(re.search(r"\b(when|what date|which date)\b", t))
    context = denial or previous in ('denial', 'appeal', 'denial_details')
    topics = []
    if denial:
        topics.append('reason')
        if re.search(r'\bappeal\b', t):
            topics.append('appeal_decision')
    if date_question and context and (denial or not notification):
        topics.append('decision_date')
    if notification:
        topics.append('notification')
    return topics


def intent_of(text, previous=None):
    t = text.lower().replace('’', "'")
    if not re.search(r"\b(not|don't|do not|never)\b", t) and re.search(r"\b(done|finished|goodbye|bye|nothing else|no more questions|all set|that's (?:all|everything)|that is (?:all|everything))\b", t):
        return 'finish'
    topics = decision_topics(t, previous)
    if topics and (any(topic != 'reason' for topic in topics) or not re.search(r'\b(documents?|paperwork|pathology|office note|missing)\b', t)):
        return 'denial_details'
    rules = (
        ('alternatives', r"\b(can't|cannot|unable|don't have|do not have|replacement|substitute|alternative)\b"),
        ('timing', r'\b(how long|how soon|processing time|review time|when.*submit)\b'),
        ('receipt', r'\b(confirm.*receipt|confirmation|got it|received.*files)\b'),
        ('format', r'\b(format|file type|scan|pdf|clear enough)\b'),
        ('submission', r'\b(upload|submit|send|portal|fax|mail)\b'),
        ('appeal', r'\b(appeal|deadline|expired|missed)\b'),
        ('documents', r'\b(documents?|paperwork|pathology|office note|missing)\b'),
        ('denial', r'\b(denial|denied|reason|rejected)\b'),
        ('payment', r'\b(paid|payment|reimbursement|net pay|amount)\b'),
        ('status', r'\b(status|progress|update)\b'),
        ('next_steps', r'\b(next steps?|what (?:can|should) i do|options)\b'),
    )
    return next((name for name, pattern in rules if re.search(pattern, t)), None)

def normalize_dob(value):
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%B %d %Y', '%b %d %Y'):
        try:
            return datetime.strptime(' '.join(value.replace(',', '').split()), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def extract_identity(text):
    """Extract only values explicitly present in this utterance."""
    identity = {}
    t = text.replace('’', "'")
    # Accept an opening verification tuple ending at a sentence boundary.
    # Later claim context must not invalidate it or become identity evidence.
    month_pattern = '|'.join(month + '|' + month[:3] for month in MONTHS)
    date_pattern = rf'(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}/\d{{4}}|(?:{month_pattern})\s+\d{{1,2}},?\s+\d{{4}})'
    compact = re.match(
        rf"\s*(?:(?P<name>[A-Za-z]+(?:[ '-][A-Za-z]+){{0,3}}?)[,;\s]+)?"
        rf'(?P<dob>{date_pattern})(?:[,;\s]+(?P<last4>\d{{4}}))?\s*(?=$|[.!?](?:\s|$))',
        t, re.I,
    )
    if compact and not re.search(r'\b(claim|filed|incident|accident|policy|payment|paid|report)\b', compact.group('name') or '', re.I):
        dob = normalize_dob(compact.group('dob'))
        if dob:
            identity['dob'] = dob
        if compact.group('last4'):
            identity['id_last4'] = compact.group('last4')
        full_name = compact.group('name') or ''
        if len(full_name.split()) >= 2:
            identity['name'] = full_name
    # Bound a name at a clause boundary rather than greedily consuming the next field.
    name = re.search(r"\b(?:my name is|name is)\s+(.+?)(?=\s+(?:and|policy|dob|ssn|phone|email|calling)\b|[,.;!?]|$)", t, re.I)
    if not name:
        name = re.search(r"\b(?:i am|i'm|this is)\s+(.+?)(?=\s+(?:and|policy|dob|ssn|phone|email|calling)\b|[,.;!?]|$)", t, re.I)
    candidate = name.group(1).strip() if name else ''
    if name and not re.search(r'\b(?:my name is|name is)\b', t, re.I) and not all(part[0].isupper() for part in candidate.split()):
        candidate = ''
    if not candidate and re.match(r'^[A-Za-z]+(?:[ -][A-Za-z]+){1,3}\s*,\s*\d{4}-', t):
        candidate = t.split(',')[0].strip()
    if re.fullmatch(r"[A-Za-z]+(?:[ '-][A-Za-z]+){1,3}", candidate) and not re.search(r'\b(not|upset|worried|angry|calling|policyholder|confused|done|terrified|very|refusing)\b', candidate, re.I):
        identity['name'] = candidate
    date_match = re.search(r'\b(?:dob|date of birth|born(?: on)?|birthday)\s*(?:is|:)?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|[A-Za-z]+ \d{1,2},? \d{4})', t, re.I)
    if not date_match:
        date_match = re.search(r'\b\d{4}-\d{2}-\d{2}\b', t) if re.fullmatch(r'[A-Za-z ,\d-]+', t) and not re.search(r'claim|filed|incident', t, re.I) else None
    if date_match:
        value = date_match.group(1) if date_match.lastindex else date_match.group()
        dob = normalize_dob(value)
        if dob:
            identity['dob'] = dob
    patterns = {
        'phone': r'\b(?:phone|mobile)(?: number)?\s*(?:is|:)?\s*(\+?\d[\d ()-]{8,}\d)',
        'id_last4': r'\b(?:ssn|social security|national id|id)\s*(?:(?:last\s*(?:four|4)(?:\s*digits)?)|(?:ends?|ending)\s*in)?\s*(?:is|:)?\s*(\d{4})\b',
        'email': r'\b([\w.+-]+@[\w.-]+\.[a-z]{2,})\b',
        'policy_number': r'\b(POL-\d+)\b',
    }
    for field, pattern in patterns.items():
        match = re.search(pattern, t, re.I)
        if match:
            identity[field] = match.group(1)
    last4 = re.search(r'\blast\s*(?:four|4)\s*(?:digits)?\s*(?:is|are|:)?\s*(\d{4})\b', t, re.I)
    if last4:
        identity['id_last4'] = last4.group(1)
    if re.fullmatch(r'(?:[A-Za-z]+(?:[ -][A-Za-z]+){1,3},\s*)?\d{4}-\d{2}-\d{2}\s*[,;]\s*\d{4}\s*', t):
        identity['id_last4'] = t.strip()[-4:]
    return identity

class RuleInterpreter:
    def interpret(self, text: str, state=None):
        t = text.replace('’', "'")
        identity = extract_identity(t)
        # A full-name answer can complete an earlier compact partial answer.
        if (state and state.phase == Phase.VERIFY_ID and 'name' not in state.identity
                and {'dob', 'id_last4'} <= state.identity.keys()
                and re.fullmatch(r"[A-Za-z]+(?:[ '-][A-Za-z]+){1,3}", t.strip())
                and not re.search(r"\b(no|not|thanks|thank|please|help|angry|upset|worried|refuse|don't|skip|human|representative)\b", t, re.I)):
            identity['name'] = t.strip()
        hint_text = t
        if identity.get('dob'):
            hint_text = re.sub(
                r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b',
                lambda match: '' if normalize_dob(match.group()) == identity['dob'] else match.group(),
                t,
            )
        hint = Hint()
        case = re.search(r'\bCL-\d+\b', t, re.I)
        if case:
            hint.case_id = case.group().upper()
        for kind in ('healthcare', 'dental', 'auto'):
            if re.search(rf'\b{kind}\b', t, re.I):
                hint.case_type = kind
        for status in ('denied', 'closed', 'open'):
            if re.search(rf'\b{status}\b', t, re.I):
                hint.status = status
        if 'dob' not in identity or re.search(r'\b(claim|filed|incident)\b', t, re.I):
            for number, month in enumerate(MONTHS, 1):
                if re.search(rf'\b{month[:3]}(?:{month[3:]})?\b', hint_text, re.I):
                    hint.month = number
            year = re.search(r'\b(?:claim|filed|from|in)\s+(?:\w+\s+)?(20\d{2})\b', t, re.I)
            if year:
                hint.year = int(year.group(1))
        intent = intent_of(t, state.current_intent if state else None)
        if state and state.phase == Phase.POST_PROCESS:
            intent = consent_of(t) or (intent if intent in BUSINESS_INTENTS else None)
        emotion = bool(re.search(r'\b(ridiculous|frustrated|angry|upset|unfair|anxious|worried|stressed|uncomfortable|confused|confusing|terrified|scared|already told you)\b', t, re.I))
        refusal = bool(re.search(r"\b(refuse|refusing|won't provide|will not provide|don't want to (?:give|share|provide)|do not want to (?:give|share|provide))\b", t, re.I))
        human = bool(re.search(r'\b(human|representative|supervisor|real person|live agent|transfer me)\b', t, re.I)) and not re.search(r"\b(no|don't|do not)\b", t, re.I)
        related = bool(identity or hint.model_dump(exclude_none=True) or intent or emotion or refusal or human or re.search(r'\b(insurance|claim|verification|identity|consent|email|hello|hi|thanks|thank you|yes|no|okay|ok|sure|why|help|correct|forget|remove)\b', t, re.I))
        explicit_offtopic = bool(re.search(r'\b(reinforcement learning|what is rl|quantum mechanics|neural network|python tutorial|write code|weather|sports scores?|recipe|joke|capital of france)\b', t, re.I))
        return Parsed(identity=identity, hint=hint, intent=intent, emotion='distressed' if emotion else None,
                      refusal=refusal, human=human, unrelated=explicit_offtopic or not related)

