# SOP-Guided Conversational Agent

The harness owns verification, claim access, phase transitions, and consent. In model mode, a separate conversational layer understands the entire message and recent history, recognizes implied emotion, and selects evidence-backed answers and adds emotional framing. Rules mode is a limited fallback, not equivalent natural-language coverage.

## Run

`docker compose up --build -d`, then open http://localhost:8000.
The default without configuration is rules mode. Set the model environment below to enable conversational responses. The UI shows conversational, basic rules, or limited/fallback mode per response. A configured provider is not proof of availability.

Tests: `docker compose exec -T -e LLM_MODE=rules -e SESSION_DB=:memory: agent python -m pytest -q -p no:cacheprovider`.
Opt-in live evaluation (uses the configured model): `docker compose exec -T -e RUN_LIVE_EVAL=1 -e SESSION_DB=:memory: agent python -m pytest -q -s tests/test_live_conversation.py -p no:cacheprovider`.
Live checks never count a rule fallback as success. Inspect printed answers for semantic correctness; neither a passing model reviewer nor a small test set proves universal reliability.

## Model configuration

Set these in `.env`, then recreate the container:

```
LLM_MODE=model
OPENAI_API_KEY=your-token
OPENAI_MODEL=gpt-4o-mini
LLM_JSON_SCHEMA=true
```

For your independently installed local Ollama server instead:

```
LLM_MODE=model
LLM_API_KEY=ollama
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen3:14b-q4_K_M
LLM_REASONING_EFFORT=none
LLM_TIMEOUT_SECONDS=60
LLM_JSON_SCHEMA=true
```

Start the existing server with `ollama serve` if it is not running. A local model must already be installed; this project does not download one. Host access shown above is for Docker Desktop. Other Docker hosts may need a host-gateway configuration.

`LLM_API_KEY` and `LLM_MODEL` override the OpenAI equivalents. A provider lacking JSON-schema support can use `LLM_JSON_SCHEMA=false`; responses still undergo local schema validation. Reasoning effort is omitted unless configured. Requests have no automatic provider retries, a bounded 5–60 second timeout per call, and a short failure cooldown. Up to three calls form a generated response; local model latency can be substantial. The UI allows 190 seconds before reporting a timeout.

Do not assume a small model is sufficient: the locally available Qwen2.5 3B produced unsupported appeal/notification statements in evaluation and its self-review missed them. Neither it nor the larger local model is trusted to write factual claim statements freely; this finding motivated the evidence-bound renderer. Cloud-provider quota errors trigger visible fallback rather than a silent capability downgrade.

## Architecture and control boundaries

1. **Local extraction and state:** Identity values are extracted from caller text and compared deterministically with synthetic fixtures. Three distinct matching fields are required. Full name, DOB, phone, email, and last four ID digits count; policy number does not. No model output can set identity or verification.
2. **Semantic understanding:** A schema-validated model plan contains all user questions, scope, case hints, emotion/intensity with a supporting quote, and requests to finish or get human help. The planner receives minimized message text and bounded recent history, not the policyholder database. Model hints cannot authorize access to another customer's claim.
3. **Harness actions:** Only legal phase transitions occur. A model suggestion to finish first asks for confirmation; a human-help suggestion offers a handoff rather than claiming a transfer. Email remains an explicit local send/skip decision. Qualified consent triggers clarification.
4. **Authorized evidence:** Only after verification and owned-claim selection does the composer receive that claim, the applicable document guidance, current workflow state, and explicit data limitations. No user assertion or model inference becomes a claim fact.
5. **Response composition and review:** The model selects and orders approved fact statements and chooses a follow-up, with a short generated emotional acknowledgment. The harness renders factual statements verbatim from authorized data; the model cannot rewrite a claim denial as an appeal denial or invent a notice. Unsupported factual-sounding acknowledgment text is removed. A separate model call reviews coverage and behavior. Unknown evidence IDs, invalid output, provider errors, and rejected plans return a guarded fallback. Review and emotional interpretation remain probabilistic; production use needs broader evaluation.

VERIFY_ID → RESOLVE_INTENT → PROCESS_CASE → POST_PROCESS → COMPLETE remains the only normal phase path. Human escalation is recorded separately without falsely completing the SOP.

## Emotion and recovery

Model mode recognizes frustration, anger, anxiety, sadness, confusion, and overwhelm from meaning and context, not only emotion keywords. An emotion inference needs a quote from the current minimized message. Intensity 3 or three consecutive turns at intensity 2+ offers human support. Distress alone is not refusal. The verification gate remains intact. Before verification, emotion selects a brief acknowledgment followed by the required local verification prompt; private claim details never enter the composer. After verification, the composer can use a situation-specific acknowledgment and select multiple approved statements for multiple questions. Factual wording stays controlled; it is not an unrestricted chatbot.

The last six exchanges and unresolved questions are retained for follow-ups. Identity values recognized by local extraction and number/email patterns are minimized before model input/history. This is best-effort minimization, not general PII anonymization. Use synthetic information only. The configured provider receives minimized messages and, after verification, claim information and approved guidance. Previous history is not authoritative evidence.

## Example

1. `Margaret March 15 1985 4472`
2. `Margaret Chen`
3. `Why is my appeal denied when did this happen why was I not notified for it?`
4. `I have been going round in circles and nobody tells me anything.`
5. `done`
6. `send` or `skip`

The record contains a claim-denial reason, not an appeal decision. The filing date and appeal deadline are not a denial date. Notification history is unavailable. The agent should acknowledge the concern and state these limits rather than inventing a notice, date, or appeal outcome.

## Storage and limits

SQLite persists Docker sessions in the `sessions` volume. Local Python uses memory unless `SESSION_DB` names a writable file. Sessions and queued summaries expire after 30 minutes; the next turn purges expired rows. Identity values are cleared on successful verification or accepted handoff. History is bounded to 12 messages and discussion notes to 20 entries. Model calls happen outside write transactions; revision checks prevent concurrent turns overwriting newer state. Email queuing and state commit are atomic, with one outbox entry per session.

`/api/outbox` is disabled unless `ADMIN_API_TOKEN` is configured; then it requires a Bearer token. The server binds to localhost. Session IDs are bearer capabilities and must not be shared. Keep `.env` private; it is excluded from Docker context and Git.

This remains a synthetic demonstration. No real email, live transfer, claim mutation, representative authorization, or production identity proofing is provided. Public hosting requires real authentication, TLS, rate limits, audited retention/encryption, and operational monitoring. Natural-language and emotional interpretation are fallible; uncertain cases should clarify or offer human help.

## Files

- `conversation.py`: provider adapter, validated understanding, composition, review, and minimization.
- `workflow.py`: authority and transition gates, authorized evidence, recovery, persistence coordination.
- `interpreter.py`: deterministic identity/consent parsing and limited offline fallback.
- `models.py` / `store.py`: typed state, bounded history, revision checks, SQLite outbox.
- `main.py` / `index.html`: API and UI with visible capability status and serialized submissions.

Provider references: [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Ollama OpenAI compatibility](https://github.com/ollama/ollama/blob/main/docs/api/openai-compatibility.mdx).
