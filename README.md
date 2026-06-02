# Origin Referral Inbox Agent

A triage agent for Cedar Kids Therapy's shared inbox.

It reads referrals, voicemails, portal messages, and emails, then produces one structured review item for each message. The agent can classify the message, pull out intake details, call the provided tools, create staff tasks, and draft a reply for human review.

## Run it

```bash
npm install
cp .env.example .env
npm run triage
npm run validate
```

Add `ANTHROPIC_API_KEY` to `.env` if you want Claude to handle extraction and classification. Without the key, the app uses a simpler rule-based fallback so it can still run.

Useful commands:

```bash
npm run typecheck
npm run eval
```

## How it works

The project is split into two parts:

1. Claude reads each inbox item and returns structured facts, like child name, insurance, requested service, language, and message type.
2. TypeScript code decides what happens next, such as which tool to call, how urgent the case is, which staff queue gets a task, and what the draft reply should say.

That split keeps the fuzzy reading work with the model and the safety-sensitive decisions in regular code.

```
inbox item
  -> Claude extracts and classifies
  -> code routes the item
  -> tools run
  -> structured output for human review
```

## Important files

- `src/index.ts` starts the batch run.
- `src/llm.ts` sends one inbox item to Claude and checks that the response has the expected shape.
- `src/agent.ts` routes each item and handles the policy logic.
- `src/tools.ts` contains the provided tools and records the audit trail.
- `src/types.ts` defines the shared data shapes.
- `src/validate.ts` checks the output against the schema.
- `scripts/run-eval.ts` runs the extra quality checks.

## Rules the agent follows

- Safeguarding messages are escalated as P0 and routed to the clinical lead.
- Same-day scheduling issues are P1 and routed to the front desk.
- New referrals check age, service line, insurance, and available slots before drafting a reply.
- Clinical questions do not get clinical advice in the draft.
- Every item still requires human review.

## Evaluation

The built-in validator checks the output format and tool-call trail. The eval script adds quality checks for classification, urgency, safeguarding handling, draft safety, and whether the right tools were called.

## What I left out

- A full agent loop where Claude decides which tools to call. I kept tool routing in code so safeguarding and policy steps stay predictable.
- LLM-written draft replies. The current drafts are templated to avoid clinical advice or accidental scheduling promises.
- PDF parsing. The sample data references attachments, but the actual PDFs are not included.
- Slot holding. The agent finds possible slots for staff review, but does not reserve anything before a family confirms.
- Retry logic for Claude calls. If Claude is unavailable, the app falls back to simpler rule-based extraction.

## With another 4 hours

- Expand the eval set with more difficult safeguarding, scheduling, insurance, and language cases.
- Add LLM-written drafts behind a safety filter, with templates as the fallback.
- Add a confidence score so uncertain items can be routed to human classification sooner.
- Track which policy rule shaped each decision, so reviewers can see why the agent acted.
- Add an evidence check where the agent points to the exact message text behind each important decision.
- Parse referral attachments if the actual files are available.
