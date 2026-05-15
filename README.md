# Origin AI Engineering Take-Home: Referral Inbox Triage Agent

Triage agent for Cedar Kids Therapy's Monday inbox. It reads a batch of mixed-channel items (faxes, voicemail transcripts, portal messages, emails), uses the provided tools where they help the decision, and emits one structured output per item for human review. Claude does the soft judgment (extraction + classification); deterministic code owns tool orchestration, policy enforcement, and the audit trail.

## How to run

```bash
npm install
cp .env.example .env       # add your ANTHROPIC_API_KEY
npm run triage   -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
npm run validate -- --input data/inbox.json --output output.json --trace .trace/tool-calls.jsonl
```

Both commands also run with no flags and default to the paths above. The agent **falls back to deterministic regex extraction if `ANTHROPIC_API_KEY` is not set**, so a reviewer can run without the key (the output is slightly less polished but still passes validation).

Optional eval harness:
```bash
npm run eval:generate      # generates 15 synthetic inbox items via Claude
npm run eval               # runs the agent against hand-labelled gold cases and prints metrics
```

`npm run typecheck` runs `tsc --noEmit`. End-to-end runtime on the 8 visible items is around 5-15 seconds (one Claude call per item with a cached system prompt).

## Stack and runtime

- TypeScript on Node LTS, `tsx` to run sources directly
- `@anthropic-ai/sdk` (`claude-opus-4-7`) for extraction + classification, via the SDK's `messages.parse()` + Zod schema for structured output
- Anthropic prompt caching on the system prompt (same across all items, so the cache amortizes across the batch)
- Deterministic handlers downstream of the LLM call for tool routing, policy enforcement, and draft templates
- Other deps: `ajv` (schema validation, provided), `dotenv` (env loading), `zod` (schema for structured output)

## Architecture

```
inbox item
  ↓
[LLM call: extract + classify]   ← Claude Opus 4.7, single call per item
  ↓ returns { intake, classification, safeguarding_reason, is_same_day, language, first names }
  ↓
[deterministic dispatch + handlers]   ← code, not LLM
  ↓ tools called via withItemContext
  ↓
ItemOutput
```

The LLM owns extraction and classification, which are the parts that benefit from understanding paraphrased language. Everything downstream (which tools to call, urgency, escalation severity, draft templates) is code. That keeps the safety-critical pieces inspectable and the tool-call audit trail deterministic.

**Key pieces:**
- [src/agent.ts](src/agent.ts): `runAgent` wraps each item in `withItemContext`, calls `llmExtractAndClassify` (with a deterministic regex fallback), then dispatches to one of five handlers based on classification.
- [src/llm.ts](src/llm.ts): single function `llmExtractAndClassify`. System prompt embeds the policy text and a classification rubric. Output is validated against a Zod schema, so a malformed response throws and the agent falls back to regex.
- [src/tools.ts](src/tools.ts): unchanged from the starter.
- [scripts/generate-eval-set.ts](scripts/generate-eval-set.ts): asks Claude to produce 15 tricky synthetic inbox items.
- [scripts/run-eval.ts](scripts/run-eval.ts): runs the agent on hand-labelled cases, reports per-axis metrics, and exits non-zero on P0 recall regression or hard-rule draft failure.

**Per-handler logic:**
- **Safeguarding** (P0): `escalate` + `lookup_policy(safeguarding)` + `create_task(clinical_lead, same-hour)` + neutral acknowledgement draft. No investigative content in the draft.
- **Scheduling**: P1 if same-day (the LLM sets the flag), P2 otherwise. `search_patient` lookup when name+DOB present. `lookup_policy(cancellation)` for same-day. When discipline is known, `find_slots` is also called to pre-pull candidate slots for the front desk (per policy: agents may find slots for human review, must not schedule).
- **Referral**: Two service-line guards run first.
  - **Age 0-18 check**: if the patient is over 18, agent looks up service-line policy, creates a front-desk task, drafts a polite decline. No insurance verification, no slot search.
  - **Discipline-in-scope check**: if the requested service isn't SLP / OT / PT (e.g. ABA, psychology, audiology), same decline pattern with a draft that names what we do offer.
  - Otherwise: `verify_insurance` then branches on result. Out-of-network or expired → billing task, lookup insurance policy, no slot search; family draft surfaces the discrepancy by name. In-network → `find_slots` (with Spanish filter if needed) + intake task; if the plan flags `auth_required`, the family draft mentions the prior-auth step. After `find_slots`, the agent **filters the returned slots against each provider's `age_range`** (read from `data/providers.json`); if no returned provider covers the child's age, the intake task notes it so staff can escalate to clinical lead.
  - **Guardian mismatch** (existing patient on file with a different guardian than the message sender) interrupts the in-network happy path and routes to front desk for verification before any patient info is shared.
  - For **all fax referrals**, the agent also drafts a phone-callback acknowledgement to the referring pediatrician's office with a status update (in-network proceeding / OON pending billing / out-of-age / out-of-scope / missing fields) and the attachment count.
- **Clinical question**: no clinical advice in the draft; offers a screening pathway; intake task.
- **Missing paperwork**: pediatrician acknowledgement with the list of missing fields (human-readable); intake task to chase the referring practice; no family draft.
- **Other** (billing, complaint, provider_followup, spam): routed to the appropriate staff queue. Spam is P3, the rest are P2.

Every item gets `requires_human_review: true`. The agent is a triage layer, not an executor.

## Quality, not just structure

The provided `npm run validate` only checks structural things (schema, audit trail, tool threshold). I added an eval harness in [scripts/run-eval.ts](scripts/run-eval.ts) that checks the four quality dimensions the validator doesn't:

1. **Triage decisions** — classification accuracy and urgency accuracy against hand-labelled gold
2. **Drafts** — a hard-rule scan for forbidden phrases (clinical advice, false scheduling commitments, safeguarding-acknowledgement language) plus an LLM-as-judge rating each draft 1-3 on on_policy / tone / specificity against the policy text
3. **Safeguarding** — P0 recall and P0 precision tracked separately, since missing a real disclosure and over-escalating are both production failure modes
4. **Tools called for the right reasons** — per-case `must_call_tools` / `must_not_call_tools` checks

The eval set is hybrid: Claude generates the inbox items (covering paraphrased safeguarding, hard negatives, mixed clinical+scheduling, indirect OON payers, same-day without "today", partial missing paperwork, Spanish with English code-switching, etc.), but the gold labels are hand-written. That split is what makes the eval honest — if Claude wrote both halves, we'd be measuring whether the agent matches Claude's biases.

Latest results on the 17 eval cases (15 originals + 2 for the age-range policy):

```
Triage decisions
  classification accuracy:  17/17 (100%)
  urgency accuracy:         17/17 (100%)
  both correct:             17/17
Safeguarding correctness
  P0 recall:                2/2
  P0 precision:             2/2 (no false positives, including the lawnmower hard negative)
Tool-call correctness
  must-call coverage:       17/17
  must-not-call clean:      17/17
Draft quality (hard rules)
  no forbidden phrases:     17/17
Draft quality (LLM judge, n=12)
  on_policy avg:            2.3 / 3
  tone avg:                 2.5 / 3
  specificity avg:          2.2 / 3
```

The judge sees 12 of 17 drafts; the other 5 are categories where the agent intentionally writes no draft (missing paperwork, existing-patient records request, billing, complaint, spam). The ~2.2 specificity score is the most honest finding — drafts use first names but otherwise lean on templated phrasing; an LLM-written draft (with a clinical-advice filter) would score higher here.

The two age-range cases verify:
- **eval_16**: 19-year-old referral. Agent calls `lookup_policy(service_lines)`, creates a front-desk task to send a polite decline, drafts a message that names the 0-18 service line. Insurance verification and slot search are deliberately not called.
- **eval_17**: 16-year-old SLP referral. Insurance verifies in-network, `find_slots` returns 4 slots from SLPs who only serve up to age 8 or 12. The intake task notes: *"4 candidate slots returned by find_slots, but none cover age 16; staff to escalate to clinical lead about coverage."*

## Failure modes and production eval

- **Safeguarding recall is the highest-stakes axis.** The LLM catches paraphrased disclosures the regex misses (`"His dad has been drinking more than usual"` + `"marks on his upper arm"` + `"freezing up when he hears the truck pull in"`). Production version would expand the eval set to ~50 cases and run it on every prompt change, gating the merge on P0 recall = 1.0.
- **Over-escalation** is also a failure mode and the rubric calls it out. The eval includes a hard negative ("husband was being rough with the wrench") to keep precision honest.
- **Insurance source-of-truth conflicts**: the policy says verified status from the billing system supersedes the referral doc. The agent already trusts `verify_insurance`, and the draft surfaces the discrepancy by name ("Our billing system is showing the X coverage as expired") so the family isn't blindsided.
- **Guardian mismatch**: caught by `search_patient` returning a record whose `guardian_name` doesn't overlap the message sender. In production I'd log this as a structured event with sender, guardian-on-file, and patient ID for trend analysis (false-positive rate, legitimate name changes).
- **Language detection**: the LLM is more reliable than my regex for short messages with mixed languages. The eval set includes a Spanish-with-English-loanwords case (`"insurance nuevo"`, `"speech therapist"`) and the agent picks Spanish correctly.
- **Tool-call hygiene**: the validator and the eval's `must_call_tools` / `must_not_call_tools` together gate every dispatch path. Any future agent rewrite has to pass both.

## What I chose not to build, and why

- **Full tool-use agent loop** (LLM decides which tools to call). More impressive on paper, but the LLM could skip an `escalate` on a paraphrased safeguarding case. The rubric weights safety judgment at 25%, so the downside is bigger than the upside without a much larger eval set. The current architecture lets me upgrade to this in a follow-up without rewriting the audit/policy layer.
- **LLM-generated draft replies**. The warmth pass with first-name extraction gets most of the readability benefit. Letting the LLM write drafts opens a clinical-advice / false-promise failure mode that's hard to fully test in 2 hours. The safer version is "LLM drafts → deterministic filter for forbidden phrases → fall back to template on hit," which is more code than I had budget for.
- **PDF parsing**. Attachments are referenced by filename only; bodies already carry the structured fields.
- **`hold_slot` calls**. Holding a slot without family preference confirmation is the kind of overreach the rubric flags. Item 1's stated preference ("after school Tues/Thurs") doesn't match any returned slot, and item 7 gives no preference at all. Better to let intake call before holding.
- **Retries / backoff for the LLM call**. The deterministic fallback path covers transient failures end-to-end, which is the production-safe behavior. Adding retries would be more code without much practical gain at this scope.

## What I would do with another 4 hours

1. **Grow the eval set to ~50 cases**, with adversarial safeguarding paraphrases that I write specifically to break the current keyword + LLM combo (e.g. "Dad's been having a hard time and the kids have been sleeping at my house all week").
2. **Move drafts to LLM-generation behind a filter**. LLM writes the draft; a deterministic check scans for clinical-advice phrases and scheduling commitments; on a hit, fall back to the current template. Judge specificity should climb from 2.0 toward 2.7+.
3. **Confidence + abstain** on classification. The LLM returns an explicit confidence; below threshold the agent abstains from drafting and routes to "needs human classification" instead.
4. **Provenance in decisions**. When a handler is shaped by a specific policy snippet, attach the snippet ID to the decision rationale so a reviewer can verify the agent is citing real policy, not confabulating.
5. **Tool-use agent prototype** as a side-by-side experiment, gated by the eval. Specifically interesting for the referral path where the order of tool calls (search_patient before verify_insurance, or vice versa) varies by case.
6. **Attachment parsing**. Inbox items reference attachments by filename only (e.g. `referral_item_1.pdf`); the actual PDFs aren't in the repo. If they were, I would add a preprocessing step that OCRs each attachment and feeds the extracted text into the LLM prompt alongside the message body, so the model can pull DOB, payer, member ID, and clinical narrative directly from the referral document rather than relying only on what the fax cover sheet repeats. For now the agent surfaces the attachment *count* in the pediatrician acknowledgement draft so a human reviewer knows to grab them when processing.
7. **Discipline-not-offered check**. The LLM extracts discipline as one of `SLP / OT / PT` or `null`. The agent doesn't currently distinguish "couldn't determine discipline" from "discipline requested isn't in our service line" (e.g. ABA, psychology). Adding a free-text `requested_service` field to the extraction, plus a deterministic check that maps it to our three offerings, would close that gap the same way the age check does for the 0-18 service line.

## Triage decisions on the visible batch

| Item | Classification | Urgency | Why |
| --- | --- | --- | --- |
| 1 Emma Lee | new_referral | P2 | BCBS in-network, clean SLP referral, slots searched, intake task, warm draft to Daniel |
| 2 Maria Gomez | safeguarding | P0 | LLM caught "dad started getting rough with him during weekends"; escalate + clinical-lead task + neutral acknowledgement |
| 3 Owen Brooks | new_referral | P2 | Kaiser out-of-network; billing task, OON draft, no slot search |
| 4 Mateo Ramirez | new_referral | P2 | Aetna in-network but `search_patient` returns existing record with a different guardian (Sofia vs sender Carla) — front desk to verify |
| 5 Jordan Kim | clinical_question | P2 | Asks for advice on R sounds; draft offers a screening, no clinical advice |
| 6 Sam Taylor | missing_paperwork | P2 | Referral has DOB / parent / insurance blank; intake to contact referring practice; no family draft |
| 7 Ana Lopez | new_referral | P2 | Medicaid in-network, Spanish-speaking; slots filtered to Spanish provider; draft and policy in `language_access` |
| 8 Anita Patel | scheduling | P1 | Same-day reschedule for existing patient; `search_patient` confirms, cancellation policy consulted, front-desk task today |
