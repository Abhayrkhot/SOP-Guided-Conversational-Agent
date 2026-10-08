import json
from pathlib import Path

class Repository:
    def __init__(self):
        root = Path(__file__).resolve().parents[1] / 'fixtures'
        self.people = json.loads((root / 'policyholders.json').read_text())
        self.claims = json.loads((root / 'claims.json').read_text())
        self.guidance = json.loads((root / 'required_document_guideline.json').read_text())

    def verify(self, identity):
        fields = ('name', 'dob', 'phone', 'email', 'id_last4')
        if len(set(identity) & set(fields)) < 3:
            return None
        def norm(value, field):
            if field == 'phone':
                return ''.join(x for x in str(value) if x.isdigit())[-10:]
            return ' '.join(str(value).casefold().split())
        matches = []
        for person in self.people:
            if 'policy_number' in identity and norm(identity['policy_number'], '') != norm(person['policy_number'], ''):
                continue
            correct = 0
            for field in fields:
                if field not in identity:
                    continue
                values = [person.get(field, '')] + person.get(field + '_aliases', [])
                if any(norm(identity[field], field) == norm(v, field) for v in values):
                    correct += 1
                else:
                    break
            else:
                if correct >= 3:
                    matches.append(person)
        return matches[0] if len(matches) == 1 else None

    def claims_for(self, party_id, hint):
        found = [c for c in self.claims if c['party_id'] == party_id]
        if hint.case_id:
            return [c for c in found if c['case_id'] == hint.case_id]
        for field in ('case_type', 'status'):
            value = getattr(hint, field)
            if value:
                found = [c for c in found if c[field] == value]
        if hint.month:
            found = [c for c in found if int(c['created_at'][5:7]) == hint.month]
        if hint.year:
            found = [c for c in found if int(c['created_at'][:4]) == hint.year]
        return found

    def claim(self, party_id, case_id):
        return next((c for c in self.claims if c['party_id'] == party_id and c['case_id'] == case_id), None)

    def person(self, party_id):
        return next(p for p in self.people if p['party_id'] == party_id)
