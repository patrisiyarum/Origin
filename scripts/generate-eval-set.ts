import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import dotenv from "dotenv";
import { z } from "zod";

dotenv.config({ override: true, quiet: true });

const InboxItemSchema = z.object({
  id: z.string(),
  channel: z.enum(["fax_referral", "voicemail_transcript", "portal_message", "email"]),
  received_at: z.string(),
  sender: z.string(),
  subject: z.string(),
  body: z.string(),
  attachments: z.array(z.string()),
  scenario_tag: z.string(),
});

const ResponseSchema = z.object({ items: z.array(InboxItemSchema).min(12).max(20) });

const PROMPT = `You are generating synthetic test cases for an inbox-triage agent at Cedar Kids Therapy, a pediatric therapy practice (SLP, OT, PT). Produce realistic but fictional inbox items.

Each case must:
- Use fully synthetic data — invented names that are not real public figures, phone numbers in the 555-XXXX reserved range, emails on @example.com
- Match the schema: id ("eval_1" .. "eval_N"), channel, received_at (ISO 8601), sender, subject, body, attachments (array of strings, can be empty), scenario_tag (one of the tags below)
- Be 80-220 words long in the body, in natural prose

Produce 15 cases that cover these tricky scenarios (one per tag, plus a few extras):

1. "safeguarding_paraphrased" — Caregiver-perpetrated harm signal that does NOT use the words abuse, hit, hurt, neglect, or rough. Phrase it indirectly (drinking, anger, fear, withdrawn behavior, unexplained bruises). Embed inside an otherwise routine request.
2. "safeguarding_hard_negative" — Looks like safeguarding but isn't. Examples: a parent describing the dad being rough with an object, a child being clumsy, generic family stress, parent venting about commute.
3. "clinical_plus_scheduling" — Single message mixes a clinical question (developmental norms) AND a scheduling request. Strong solutions classify by the dominant intent and route appropriately.
4. "oon_indirect_payer" — Out-of-network insurance described indirectly ("Kaiser through my husband's job", "we have BeaconCare from work").
5. "expired_coverage" — Family lists a payer that the system will flag as expired (Sunrise Health, Pediatric Choice, Community First).
6. "same_day_no_today" — Same-day cancellation phrased without the word "today" ("running late, can't make our 3pm", "we'll have to miss our morning appointment").
7. "missing_partial" — Fax referral with SOME fields blank but not all (e.g. DOB present, payer blank, parent contact filled).
8. "spanish_mixed" — Family message in Spanish with some English code-switching (single sentence, common nouns left in English).
9. "existing_patient_control" — Existing patient request (records, scheduling, billing question) that is NOT a guardian mismatch — a clean existing-patient case.
10. "non_urgent_reschedule" — Reschedule that is several days out, not same-day; should be P2 not P1.
11. "billing_question" — Existing patient's family asking about a copay, EOB, or invoice.
12. "in_network_clean" — Straightforward in-network referral (Aetna, BCBS, UnitedHealthcare, Medicaid). Vary the discipline.
13. "complaint" — A polite complaint about wait time, no clinical or safety implications.
14. "spam_or_other" — Off-topic vendor pitch, sales email, or marketing message.
15. "voicemail_safeguarding" — Voicemail transcript with a soft safeguarding signal (less explicit than case 1; the system should still catch it).

Mix channels across the 15 cases (faxes from pediatric clinics, voicemails from parents, portal messages, emails). Vary discipline (SLP, OT, PT). Include at least one Spanish case and one bilingual case. Keep dates within the last 7 days from 2026-05-15.

Return JSON only.`;

async function main(): Promise<void> {
  if (!process.env.ANTHROPIC_API_KEY) {
    console.error(
      "ANTHROPIC_API_KEY not set. Set it in .env (see .env.example).",
    );
    process.exit(1);
  }

  const client = new Anthropic();
  console.error("Generating eval items...");

  const response = await client.messages.parse({
    model: "claude-opus-4-7",
    max_tokens: 16000,
    messages: [{ role: "user", content: PROMPT }],
    output_config: { format: zodOutputFormat(ResponseSchema) },
  });

  const parsed = response.parsed_output;
  if (!parsed) {
    throw new Error("LLM returned no parseable output");
  }

  const outputPath = resolve(process.cwd(), "data/eval/raw_items.json");
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, `${JSON.stringify(parsed.items, null, 2)}\n`);

  console.error(`Wrote ${parsed.items.length} items to ${outputPath}`);
  console.error(
    `Next: copy raw_items.json to cases.json and hand-add an 'expected' block per case.`,
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
