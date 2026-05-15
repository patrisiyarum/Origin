import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";
import type { Classification, Discipline, InboxItem } from "./types.js";

const ExtractionSchema = z.object({
  child_name: z.string().nullable(),
  dob_or_age: z.string().nullable(),
  parent_contact: z.string().nullable(),
  discipline: z
    .array(z.enum(["SLP", "OT", "PT"]))
    .min(1)
    .nullable(),
  diagnosis_or_concern: z.string().nullable(),
  payer: z.string().nullable(),
  member_id: z.string().nullable(),
  classification: z.enum([
    "new_referral",
    "existing_patient_request",
    "scheduling",
    "clinical_question",
    "billing_question",
    "missing_paperwork",
    "provider_followup",
    "complaint",
    "safeguarding",
    "spam",
    "other",
  ]),
  safeguarding_reason: z.string().nullable(),
  is_same_day: z.boolean(),
  language: z.enum(["en", "es"]),
  parent_first_name: z.string().nullable(),
  child_first_name: z.string().nullable(),
  referring_doctor: z.string().nullable(),
  referring_practice: z.string().nullable(),
  service_in_scope: z.boolean(),
  requested_service: z.string().nullable(),
});

export type LlmExtractionResult = {
  intake: {
    child_name: string | null;
    dob_or_age: string | null;
    parent_contact: string | null;
    discipline: Discipline[] | null;
    diagnosis_or_concern: string | null;
    payer: string | null;
    member_id: string | null;
  };
  classification: Classification;
  safeguarding_reason: string | null;
  is_same_day: boolean;
  language: "en" | "es";
  parent_first_name: string | null;
  child_first_name: string | null;
  referring_doctor: string | null;
  referring_practice: string | null;
  service_in_scope: boolean;
  requested_service: string | null;
};

let cachedClient: Anthropic | null = null;
function getClient(): Anthropic {
  if (!cachedClient) cachedClient = new Anthropic();
  return cachedClient;
}

let cachedPolicy: string | null = null;
function getPolicy(): string {
  if (!cachedPolicy) {
    cachedPolicy = readFileSync(
      resolve(process.cwd(), "data/policies.md"),
      "utf8",
    );
  }
  return cachedPolicy;
}

function systemPrompt(): string {
  return `You are the intake triage agent for Cedar Kids Therapy, a pediatric therapy practice (SLP, OT, PT). You will receive one item from a shared inbox (fax referral, voicemail transcript, parent portal message, or email) and must return a structured JSON extraction plus a classification.

You only extract and classify. You do not call tools. Downstream deterministic code uses your output to decide which tools to run, what urgency to assign, and what task or draft to create.

==========
POLICY (verbatim, treat as ground truth)
==========
${getPolicy()}

==========
CLASSIFICATION RUBRIC
==========
Pick exactly one classification using the priority order below. If two could fit, the one higher in the list wins.

1. "safeguarding" — Any language suggesting harm, abuse, neglect, or unsafe caregiving toward the child. This includes paraphrased disclosures. Examples of signals: "Dad has been really mean since he started drinking again", "She has bruises she can't explain", "He's been rough with him", "Mom leaves them alone for hours". The word "abuse" does not have to appear. When you classify as safeguarding, set safeguarding_reason to one short sentence quoting or paraphrasing the specific signal. Hard negatives: a parent describing being rough with an object ("Dad got rough with the lawnmower"), generic frustration without harm to the child, or descriptions of the child being clumsy/active.

2. "scheduling" — Request to reschedule, cancel, move, or skip an existing appointment. Set is_same_day=true if the request is about an appointment occurring today (look for "today", "this afternoon", "this morning", a specific time today, "running late", "can't make our 3pm", or any phrasing implying today's calendar). Otherwise is_same_day=false.

3. "clinical_question" — Family is asking for clinical advice or developmental reassurance and is NOT submitting a referral or scheduling request. Examples: "Is it normal that my 4-year-old can't say her R sounds?", "Should I be worried about..." If the same message also requests a referral or scheduling, prefer "new_referral" or "scheduling" instead.

4. "missing_paperwork" — A fax referral or formal submission that has blank or missing required fields (DOB, parent contact, payer, member ID, discipline). Use this when key intake fields are explicitly blank or missing, not when the message simply doesn't mention them in prose.

5. "new_referral" — A family or pediatrician is submitting a new referral for evaluation. Most fax referrals and most parent emails about "we'd like to set up an evaluation" land here.

6. "existing_patient_request" — A current patient's family is asking for something that isn't a reschedule (e.g. records request, provider question, billing concern about an existing case).

7. "billing_question", "provider_followup", "complaint", "spam", "other" — Use when nothing above fits.

==========
EXTRACTION RULES
==========
- child_name: the child's full name as written. Null if not stated.
- dob_or_age: ISO date if a full DOB is given ("DOB: 2019-03-15" or "born 2019-03-15"); otherwise a short age phrase like "age 4" or "age 6". Null if neither is stated.
- parent_contact: a single string concatenating any of: parent full name, phone, email. Comma-separated. Pull phone and email directly from the body when possible. Null only if nothing usable is present (e.g. a fax referral with all parent fields blank). The fax sender (a clinic name) is NOT a parent contact; do not use it.
- discipline: array of "SLP" / "OT" / "PT" based on the requested or implied service. SLP = speech, articulation, language. OT = sensory, feeding, fine motor. PT = gross motor, gait, toe walking. Null if undetermined.
- diagnosis_or_concern: the clinical concern in the message ("articulation delay", "sensory processing and feeding tolerance", "toe walking and frequent tripping"). Null if not stated.
- payer: insurance plan name as written ("Blue Cross Blue Shield PPO", "Kaiser HMO", "Aetna PPO", "Medicaid"). Null if blank, missing, or not stated. Normalize common aliases: "BCBS" -> "Blue Cross Blue Shield"; "UHC" -> "UnitedHealthcare".
- member_id: insurance member ID as written. Null if blank or absent.
- safeguarding_reason: only populate when classification is "safeguarding". Otherwise null.
- is_same_day: boolean. Only meaningful for scheduling items; default false for everything else.
- language: "es" if the message body is primarily Spanish (look for "hola", "gracias", "mi hija/hijo", "tiene N anos", etc.); "en" otherwise.
- parent_first_name: the parent's first name as a single word, used for warmer drafts ("Daniel", "Maria", "Ana"). Null if no parent name is available.
- child_first_name: the child's first name as a single word ("Emma", "Isabella", "Noah"). Null if no child name is available.
- referring_doctor: the referring pediatrician's name when present in the body, formatted as it appears (e.g. "Dr. Priya Nair", "Dr. Helena Yu"). Null if no doctor is named. Typically only present on fax referrals or when a parent says "our pediatrician Dr. X".
- referring_practice: the referring practice name when present (e.g. "Northside Pediatrics", "Maplewood Pediatrics"). For fax referrals, use the practice name from the sender field if not stated in the body. Null if no practice is identifiable.
- service_in_scope: true if the requested service maps to Cedar Kids' service lines (speech-language pathology / SLP, occupational therapy / OT, physical therapy / PT). false ONLY if the requested service is clearly something we don't offer (e.g. Applied Behavior Analysis / ABA, psychology, psychiatry, mental health counseling / talk therapy, audiology, vision therapy, nutrition / dietitian, music therapy). If no specific service is requested, or the message isn't a referral, or it's ambiguous, default to true (assume in scope; let the human review).
- requested_service: free-text label of the service the family or referring provider asked for, as it appears in the message (e.g. "ABA therapy", "psychological evaluation", "speech therapy"). Null if no specific service is requested.

==========
PRIORITY REMINDERS
==========
- Recall on safeguarding matters more than precision; if there's a credible signal of caregiver-perpetrated harm toward the child, classify as safeguarding even if the message also asks for a referral or scheduling. The downstream policy code will still create the right tasks.
- Do not invent fields. If something isn't in the message, return null. The fax sender field is not a parent contact.
- Do not include the parent's relationship to the child in parent_contact (no "Mom: Maria" — just "Maria Gomez, 555-0102").
- Treat 555-XXXX phone numbers and @example.com emails as valid synthetic data.

Return only the JSON; the structured-output system will validate it against the schema.`;
}

function itemToUserMessage(item: InboxItem): string {
  return `Channel: ${item.channel}
Received: ${item.received_at}
Sender: ${item.sender}
Subject: ${item.subject}

Body:
${item.body}`;
}

export async function llmExtractAndClassify(
  item: InboxItem,
): Promise<LlmExtractionResult> {
  const client = getClient();

  const response = await client.messages.parse({
    model: "claude-opus-4-7",
    max_tokens: 2048,
    system: [
      {
        type: "text",
        text: systemPrompt(),
        cache_control: { type: "ephemeral" },
      },
    ],
    messages: [{ role: "user", content: itemToUserMessage(item) }],
    output_config: { format: zodOutputFormat(ExtractionSchema) },
  });

  const parsed = response.parsed_output;
  if (!parsed) {
    throw new Error(
      `LLM returned no parseable output for item ${item.id} (stop_reason=${response.stop_reason})`,
    );
  }

  return {
    intake: {
      child_name: parsed.child_name,
      dob_or_age: parsed.dob_or_age,
      parent_contact: parsed.parent_contact,
      discipline: parsed.discipline,
      diagnosis_or_concern: parsed.diagnosis_or_concern,
      payer: parsed.payer,
      member_id: parsed.member_id,
    },
    classification: parsed.classification as Classification,
    safeguarding_reason: parsed.safeguarding_reason,
    is_same_day: parsed.is_same_day,
    language: parsed.language,
    parent_first_name: parsed.parent_first_name,
    child_first_name: parsed.child_first_name,
    referring_doctor: parsed.referring_doctor,
    referring_practice: parsed.referring_practice,
    service_in_scope: parsed.service_in_scope,
    requested_service: parsed.requested_service,
  };
}

export function isLlmAvailable(): boolean {
  return Boolean(process.env.ANTHROPIC_API_KEY);
}
