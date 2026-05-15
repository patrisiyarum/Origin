import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { isLlmAvailable, llmExtractAndClassify } from "./llm.js";
import {
  create_task,
  draft_message,
  escalate,
  find_slots,
  getToolCallsForItem,
  hold_slot,
  lookup_policy,
  search_patient,
  verify_insurance,
  withItemContext,
} from "./tools.js";
import type {
  Channel,
  Classification,
  Discipline,
  ExtractedIntake,
  InboxItem,
  ItemOutput,
  Provider,
  Urgency,
} from "./types.js";

const SERVICE_LINE_MAX_AGE = 18;

interface RouterInput {
  intake: ExtractedIntake;
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
  source: "llm" | "fallback";
}

export async function runAgent(inbox: InboxItem[]): Promise<ItemOutput[]> {
  const outputs: ItemOutput[] = [];
  for (const item of inbox) {
    const output = await withItemContext(item.id, () => processItem(item));
    outputs.push(output);
  }
  return outputs;
}

async function processItem(item: InboxItem): Promise<ItemOutput> {
  const router = await buildRouterInput(item);
  return await dispatch(item, router);
}

async function buildRouterInput(item: InboxItem): Promise<RouterInput> {
  if (isLlmAvailable()) {
    try {
      const result = await llmExtractAndClassify(item);
      return {
        intake: result.intake,
        classification: result.classification,
        safeguarding_reason: result.safeguarding_reason,
        is_same_day: result.is_same_day,
        language: result.language,
        parent_first_name: result.parent_first_name,
        child_first_name: result.child_first_name,
        referring_doctor: result.referring_doctor,
        referring_practice: result.referring_practice,
        service_in_scope: result.service_in_scope,
        requested_service: result.requested_service,
        source: "llm",
      };
    } catch (err) {
      console.error(
        `[llm fallback] item ${item.id}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
  }
  return fallbackRouterInput(item);
}

function fallbackRouterInput(item: InboxItem): RouterInput {
  const intake = extractIntake(item);
  const classification = classify(item, intake);
  const parentName = extractParentName(item.body, item.sender);
  const drMatch = item.body.match(/Dr\.\s+([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)?)/);
  const practiceMatch =
    item.channel === "fax_referral"
      ? item.sender.replace(/\s*fax$/i, "").trim() || null
      : null;
  return {
    intake,
    classification,
    safeguarding_reason: null,
    is_same_day: isSameDayScheduling(item.body),
    language: detectLanguage(item),
    parent_first_name: firstTokenOrNull(parentName),
    child_first_name: firstTokenOrNull(intake.child_name),
    referring_doctor: drMatch ? `Dr. ${drMatch[1]}` : null,
    referring_practice: practiceMatch,
    service_in_scope: !/\babb?a\b|behavioral therapy|psychology|psychiatry|counseling|audiology|vision therapy|dietitian|nutrition|music therapy/i.test(
      item.body,
    ),
    requested_service: null,
    source: "fallback",
  };
}

function firstTokenOrNull(value: string | null): string | null {
  if (!value) return null;
  const token = value.trim().split(/[\s,]+/)[0];
  return token && /^[A-Z][a-zA-Z\-']+$/.test(token) ? token : null;
}

async function dispatch(item: InboxItem, router: RouterInput): Promise<ItemOutput> {
  switch (router.classification) {
    case "safeguarding":
      return handleSafeguarding(item, router);
    case "scheduling":
      return handleScheduling(item, router);
    case "clinical_question":
      return handleClinicalQuestion(item, router);
    case "missing_paperwork":
      return handleMissingPaperwork(item, router);
    case "new_referral":
    case "existing_patient_request":
      return handleReferral(item, router, router.classification);
    default:
      return handleOther(item, router, router.classification);
  }
}

async function handleSafeguarding(
  item: InboxItem,
  router: RouterInput,
): Promise<ItemOutput> {
  const intake = router.intake;
  const signal = router.safeguarding_reason
    ? `Signal: ${router.safeguarding_reason}`
    : "caregiver-harm signal in message body";
  const reason = `Possible safeguarding disclosure in ${item.channel} from ${item.sender}. ${signal}.`;

  await escalate({ item_id: item.id, reason, severity: "P0" });
  await lookup_policy({ topic: "safeguarding" });

  const taskResult = await create_task({
    assignee: "clinical_lead",
    title: `Same-hour safeguarding review for ${intake.child_name || "child"}`,
    due: sameHourDue(item.received_at),
    notes: `${item.channel} from ${item.sender} references possible caregiver harm. ${signal}. Per policy, do not provide investigative guidance over message. Clinical lead to review and follow up directly with family and any reporting obligations.`,
  });

  const draft = await neutralAcknowledgement(item, router);

  return buildOutput({
    item,
    intake,
    classification: "safeguarding",
    urgency: "P0",
    escalation: { reason, severity: "P0" },
    task_ids: [taskResult.data.task_id],
    draft_reply: draft,
    recommended_next_action:
      "Clinical lead to triage same-hour. Do not respond with investigative content; staff to call family directly.",
    decision_rationale:
      "Message body contains language suggesting caregiver-perpetrated harm. Safeguarding policy mandates P0 escalation, clinical-lead task, and a neutral staff-reviewed acknowledgement (no investigative messaging).",
    missing_info: missingForReferral(intake),
  });
}

async function handleScheduling(
  item: InboxItem,
  router: RouterInput,
): Promise<ItemOutput> {
  const intake = router.intake;
  const sameDay = router.is_same_day;
  const childFirst = router.child_first_name;
  const parentFirst = router.parent_first_name;

  if (intake.child_name && intake.dob_or_age && looksLikeDob(intake.dob_or_age)) {
    await search_patient({
      name: intake.child_name,
      dob: intake.dob_or_age,
    });
  } else if (intake.child_name) {
    await search_patient({ name: intake.child_name });
  }

  await lookup_policy({ topic: sameDay ? "cancellation" : "scheduling" });

  const discipline = intake.discipline && intake.discipline[0];
  let candidateSlotCount: number | null = null;
  if (discipline) {
    const slots = await find_slots({
      discipline,
      language: router.language === "es" ? "es" : undefined,
      preferences: extractPreferences(item.body),
    });
    candidateSlotCount = slots.data.length;
  }

  const taskResult = await create_task({
    assignee: "front_desk",
    title: sameDay
      ? `Same-day reschedule: ${intake.child_name || "patient"}`
      : `Schedule change request: ${intake.child_name || "patient"}`,
    due: sameDay ? todayDate(item.received_at) : nextBusinessDay(item.received_at),
    notes: `${item.sender} requests ${sameDay ? "a same-day reschedule" : "a schedule change"}. Contact: ${intake.parent_contact || "see message body"}.${candidateSlotCount !== null ? ` ${candidateSlotCount} candidate ${discipline} slot${candidateSlotCount === 1 ? "" : "s"} pre-pulled for review (per policy: agents may find slots for human review, must not schedule).` : ""}`,
  });

  const greeting = greet(router.language, parentFirst);
  const wellWishes = childFirst
    ? ` Sending well wishes to ${childFirst}.`
    : "";

  const draft = await draft_message({
    recipient: contactPreference(item, intake),
    channel: replyChannel(item),
    body: sameDay
      ? `${greeting} thanks so much for letting us know, and so sorry your little one isn't feeling well. Someone from our front desk will reach out shortly to take care of the cancellation and find another time that works.${wellWishes}`
      : `${greeting} thanks for the heads up! Someone from our team will be in touch soon to confirm the change and find a time that fits your family better.`,
    language: "en",
  });

  return buildOutput({
    item,
    intake,
    classification: "scheduling",
    urgency: sameDay ? "P1" : "P2",
    escalation: null,
    task_ids: [taskResult.data.task_id],
    draft_reply: draft.args.body as string,
    recommended_next_action: sameDay
      ? "Front desk to call family within the hour, confirm cancellation, and offer a makeup slot per provider capacity."
      : "Front desk to confirm the schedule change and propose next available options.",
    decision_rationale: sameDay
      ? "Same-day cancellation/reschedule is P1 per scheduling policy. Existing patient lookup performed; task created for front desk; family acknowledgement drafted without scheduling commitment."
      : "Non-urgent schedule change. Patient lookup performed; task created for front desk follow-up.",
    missing_info: [],
  });
}

async function handleClinicalQuestion(
  item: InboxItem,
  router: RouterInput,
): Promise<ItemOutput> {
  const intake = router.intake;
  await lookup_policy({ topic: "clinical_advice" });

  const taskResult = await create_task({
    assignee: "intake",
    title: `Follow up on clinical question from ${item.sender}`,
    due: nextBusinessDay(item.received_at),
    notes: `Family asked a clinical question via ${item.channel}. Per policy, do not provide clinical advice in writing. Offer a screening or evaluation pathway and book a callback if family is interested.`,
  });

  const greeting = greet(router.language, router.parent_first_name);
  const childRef = router.child_first_name || intake.child_name || "your child";

  const draft = await draft_message({
    recipient: contactPreference(item, intake),
    channel: replyChannel(item),
    body: `${greeting} thanks so much for reaching out about ${childRef}, and totally understandable to want some reassurance. It's a little hard for us to give helpful guidance without meeting your child first, so our intake team would love to set up a quick screening with one of our speech-language pathologists. They can answer your questions in context. Someone will follow up to walk through next steps.`,
    language: "en",
  });

  return buildOutput({
    item,
    intake,
    classification: "clinical_question",
    urgency: "P2",
    escalation: null,
    task_ids: [taskResult.data.task_id],
    draft_reply: draft.args.body as string,
    recommended_next_action:
      "Intake to call family, offer a screening or evaluation, and avoid clinical advice in any written reply.",
    decision_rationale:
      "Question asks for clinical advice on developmental norms. Policy forbids clinical advice over message; draft offers a screening pathway and a task routes to intake for follow-up.",
    missing_info: [],
  });
}

async function handleMissingPaperwork(
  item: InboxItem,
  router: RouterInput,
): Promise<ItemOutput> {
  const intake = router.intake;
  const missing = missingForReferral(intake);

  await ackReferringProvider(
    item,
    router,
    `referral is missing ${missing.join(", ") || "required fields"}; please send a completed form so we can proceed`,
  );

  const taskResult = await create_task({
    assignee: "intake",
    title: `Incomplete referral from ${item.sender}: request missing fields`,
    due: nextBusinessDay(item.received_at),
    notes: `Referral missing: ${missing.join(", ") || "key intake fields"}. Contact referring practice (${item.sender}) for completed form before proceeding with scheduling or insurance verification.`,
  });

  return buildOutput({
    item,
    intake,
    classification: "missing_paperwork",
    urgency: "P2",
    escalation: null,
    task_ids: [taskResult.data.task_id],
    draft_reply: null,
    recommended_next_action:
      "Intake to contact the referring practice for the missing fields. Do not contact the family until parent contact info is supplied.",
    decision_rationale:
      "Referral is missing the fields required to verify insurance or schedule. No parent contact is available, so no family draft is generated; task routes to intake to request a complete form.",
    missing_info: missing,
  });
}

async function handleReferral(
  item: InboxItem,
  router: RouterInput,
  classification: Classification,
): Promise<ItemOutput> {
  const intake = router.intake;
  const language = router.language;
  const childFirst = router.child_first_name;
  const childRef = childFirst || intake.child_name || (language === "es" ? "su hijo/a" : "your child");
  const greeting = greet(language, router.parent_first_name);

  const childAge = parseAge(intake.dob_or_age, item.received_at);
  if (childAge !== null && childAge > SERVICE_LINE_MAX_AGE) {
    return await handleOutOfAgeRange(item, router, classification, childAge);
  }

  if (!router.service_in_scope) {
    return await handleServiceNotOffered(item, router, classification);
  }

  const missing = missingForReferral(intake);
  const taskIds: string[] = [];
  let draftBody: string | null = null;
  const rationale: string[] = [];

  let existingMismatch = false;
  if (intake.child_name && intake.dob_or_age && looksLikeDob(intake.dob_or_age)) {
    const patient = await search_patient({
      name: intake.child_name,
      dob: intake.dob_or_age,
    });
    const first = patient.data[0];
    if (first) {
      const parentName = extractParentName(item.body, item.sender);
      if (parentName && !nameOverlaps(parentName, first.guardian_name)) {
        existingMismatch = true;
        rationale.push(
          `search_patient matched ${first.name}, but sender (${parentName}) does not match guardian on file (${first.guardian_name}); flag for human verification before any scheduling step.`,
        );
      }
    }
  }

  const insurance = intake.payer
    ? await verify_insurance({
        payer: intake.payer,
        member_id: intake.member_id || undefined,
      })
    : null;

  const inNetwork = insurance?.data.status === "in_network";
  const outOfNetwork = insurance?.data.status === "out_of_network";
  const expired = insurance?.data.status === "expired";

  const urgency: Urgency = "P2";

  if (outOfNetwork || expired) {
    await lookup_policy({ topic: "insurance" });
    await ackReferringProvider(
      item,
      router,
      outOfNetwork
        ? `${insurance?.data.plan || intake.payer || "payer"} is out-of-network for our practice; billing is contacting the family before any scheduling step`
        : `${insurance?.data.plan || intake.payer || "listed coverage"} returned expired from our billing system; reaching out to the family for current insurance`,
    );
    const billingTask = await create_task({
      assignee: "billing",
      title: `${outOfNetwork ? "Out-of-network" : "Expired-coverage"} review for ${intake.child_name || "patient"}`,
      due: nextBusinessDay(item.received_at),
      notes: `${insurance?.data.plan || intake.payer || "Payer"} returned ${insurance?.data.status} from billing verification. Per policy, a benefits conversation must happen before any slot hold or scheduling step.`,
    });
    taskIds.push(billingTask.data.task_id);

    const opener = referralOpener(item, language, childRef);
    const draft = await draft_message({
      recipient: contactPreference(item, intake),
      channel: replyChannel(item),
      body: outOfNetwork
        ? `${greeting} ${opener} Before we get started, our billing team needs to take a closer look at your ${insurance?.data.plan || "insurance"} plan since it looks like it may be out of network with us. Someone will reach out soon to walk through your options together.${childFirst ? ` We're looking forward to meeting ${childFirst}.` : ""}`
        : `${greeting} ${opener} Our billing system is showing the ${insurance?.data.plan || "listed"} coverage as expired, so a team member will reach out to confirm current insurance before we move forward. Apologies for the extra step.`,
      language,
    });
    draftBody = draft.args.body as string;
    rationale.push(
      outOfNetwork
        ? "Payer verified out_of_network; billing must hold a benefits conversation before any slot hold."
        : "Billing system shows coverage expired; surfacing discrepancy and requesting current insurance before scheduling.",
    );
  } else if (inNetwork && !existingMismatch) {
    const discipline = (intake.discipline && intake.discipline[0]) || undefined;
    const prefs = extractPreferences(item.body);
    await ackReferringProvider(
      item,
      router,
      `insurance verified in-network${insurance?.data.auth_required ? " (auth required)" : ""}; intake is contacting the family to schedule`,
    );
    const slots = await find_slots({
      discipline,
      language: language === "es" ? "es" : undefined,
      preferences: prefs,
    });

    if (language === "es") {
      await lookup_policy({ topic: "language_access" });
    }

    const ageEligibleSlots =
      childAge === null
        ? slots.data
        : slots.data.filter((slot) => providerCoversAge(slot.provider_id, childAge));
    const ageMismatch = childAge !== null && slots.data.length > 0 && ageEligibleSlots.length === 0;

    const intakeTask = await create_task({
      assignee: "intake",
      title: `Schedule evaluation for ${intake.child_name || "new patient"}`,
      due: nextBusinessDay(item.received_at),
      notes: `Insurance verified ${insurance?.data.status}${insurance?.data.auth_required ? " (auth required)" : ""}. ${slots.data.length} candidate slots returned by find_slots${ageMismatch ? `, but none cover age ${childAge}; staff to escalate to clinical lead about coverage` : "; staff to confirm preferences and complete scheduling"}.${prefs ? ` Family preferences: ${prefs}.` : ""}`,
    });
    taskIds.push(intakeTask.data.task_id);
    if (ageMismatch) {
      rationale.push(
        `Child age ${childAge} not in any returned provider's age_range; intake task flagged for coverage discussion.`,
      );
    }

    const opener = referralOpener(item, language, childRef);
    const authNote = insurance?.data.auth_required
      ? language === "es"
        ? " Su seguro requiere autorizacion previa, asi que el equipo le guiara con ese paso antes de coordinar la evaluacion."
        : " Your plan does require prior authorization, so the team will walk you through that step before the evaluation is scheduled."
      : "";
    const draft = await draft_message({
      recipient: contactPreference(item, intake),
      channel: replyChannel(item),
      body:
        language === "es"
          ? `${greeting} ${opener} Ya verificamos el seguro y un miembro del equipo se comunicara con usted en espanol pronto para encontrar un horario que les funcione bien.${authNote}${childFirst ? ` Tenemos muchas ganas de conocer a ${childFirst}.` : ""}`
          : `${greeting} ${opener} We've verified the insurance and someone from our intake team will reach out soon to find an evaluation time that fits your schedule.${authNote}${childFirst ? ` Looking forward to meeting ${childFirst}.` : ""}`,
      language,
    });
    draftBody = draft.args.body as string;
    rationale.push(
      `Payer verified in_network${insurance?.data.auth_required ? " (auth flagged)" : ""}; find_slots returned ${slots.data.length} options for ${discipline || "requested discipline"}${language === "es" ? " in Spanish" : ""}; intake to confirm and schedule.`,
    );
  } else if (existingMismatch) {
    const frontDeskTask = await create_task({
      assignee: "front_desk",
      title: `Verify guardian relationship for ${intake.child_name || "existing patient"}`,
      due: nextBusinessDay(item.received_at),
      notes: `Existing patient on file with a different guardian than the message sender. Confirm legal guardian status before sharing any patient information or scheduling.`,
    });
    taskIds.push(frontDeskTask.data.task_id);

    const draft = await draft_message({
      recipient: contactPreference(item, intake),
      channel: replyChannel(item),
      body: `${greeting} thanks for reaching out about ${childRef}. Before we get started, someone from our team will reach out to confirm a few details with you. Really appreciate your patience.`,
      language,
    });
    draftBody = draft.args.body as string;
  } else {
    const intakeTask = await create_task({
      assignee: "intake",
      title: `Review referral for ${intake.child_name || "new patient"}`,
      due: nextBusinessDay(item.received_at),
      notes: `Insurance status: ${insurance?.data.status || "not provided"}. Intake to confirm coverage, contact family, and schedule.`,
    });
    taskIds.push(intakeTask.data.task_id);
    rationale.push(
      `Insurance status ${insurance?.data.status || "unknown"} requires intake follow-up before scheduling.`,
    );
  }

  return buildOutput({
    item,
    intake,
    classification,
    urgency,
    escalation: null,
    task_ids: taskIds,
    draft_reply: draftBody,
    recommended_next_action:
      outOfNetwork || expired
        ? "Billing to discuss coverage options with the family before any slot is held."
        : existingMismatch
          ? "Front desk to verify guardian relationship before sharing patient information."
          : "Intake to confirm preferences and complete scheduling using returned slot options.",
    decision_rationale:
      rationale.join(" ") ||
      "Standard referral processed with insurance verification, slot search, and intake follow-up task.",
    missing_info: missing,
  });
}

async function handleOther(
  item: InboxItem,
  router: RouterInput,
  classification: Classification,
): Promise<ItemOutput> {
  const intake = router.intake;
  const assignee: Record<string, "front_desk" | "intake" | "billing" | "clinical_lead"> = {
    billing_question: "billing",
    complaint: "front_desk",
    provider_followup: "clinical_lead",
    spam: "front_desk",
    other: "front_desk",
  };
  const urgency: Urgency = classification === "spam" ? "P3" : "P2";
  const taskResult = await create_task({
    assignee: assignee[classification] || "front_desk",
    title: `Review ${classification.replace(/_/g, " ")} from ${item.sender}`,
    due: nextBusinessDay(item.received_at),
    notes: `${classification === "spam" ? "Filtered as spam; staff to review and discard." : "Routed to staff for review and response."}`,
  });

  return buildOutput({
    item,
    intake,
    classification,
    urgency,
    escalation: null,
    task_ids: [taskResult.data.task_id],
    draft_reply: null,
    recommended_next_action:
      classification === "spam"
        ? "Mark as spam; no response needed."
        : "Staff to review and respond per category (billing, complaint follow-up, or provider question).",
    decision_rationale:
      classification === "spam"
        ? "Item is sales/marketing outreach unrelated to patient care."
        : `Item classified as ${classification}; routed to appropriate staff queue for handling.`,
    missing_info: [],
  });
}

async function handleServiceNotOffered(
  item: InboxItem,
  router: RouterInput,
  classification: Classification,
): Promise<ItemOutput> {
  const intake = router.intake;
  const language = router.language;
  const greeting = greet(language, router.parent_first_name);
  const childRef = router.child_first_name || intake.child_name || (language === "es" ? "su hijo/a" : "your child");
  const service = router.requested_service || "the service requested";

  await lookup_policy({ topic: "service_lines" });
  await ackReferringProvider(
    item,
    router,
    `requested service (${service}) is outside our service lines (we offer SLP, OT, and PT only); not proceeding with intake`,
  );

  const taskResult = await create_task({
    assignee: "front_desk",
    title: `Out-of-scope referral for ${intake.child_name || "patient"}: ${service}`,
    due: nextBusinessDay(item.received_at),
    notes: `Referral requests ${service}, which is outside Cedar Kids' service lines (SLP, OT, PT). Front desk to send polite decline and, if possible, point the family toward a relevant resource.`,
  });

  const draft = await draft_message({
    recipient: contactPreference(item, intake),
    channel: replyChannel(item),
    body:
      language === "es"
        ? `${greeting} gracias por comunicarse con Cedar Kids Therapy. Atendemos exclusivamente terapia de habla y lenguaje (SLP), terapia ocupacional (OT) y terapia fisica (PT), asi que no podemos coordinar ${service} para ${childRef}. Un miembro de nuestro equipo le contactara con sugerencias de otros recursos.`
        : `${greeting} thanks so much for reaching out to Cedar Kids Therapy. We specialize in speech-language pathology (SLP), occupational therapy (OT), and physical therapy (PT) only, so we aren't able to provide ${service} for ${childRef}. Someone from our team will follow up with suggestions for a practice that fits.`,
    language,
  });

  return buildOutput({
    item,
    intake,
    classification,
    urgency: "P2",
    escalation: null,
    task_ids: [taskResult.data.task_id],
    draft_reply: draft.args.body as string,
    recommended_next_action:
      "Front desk to send polite decline and refer the family to a practice that offers the requested service.",
    decision_rationale: `Cedar Kids service lines are SLP, OT, and PT only per policy. Requested service (${service}) is outside our scope; no insurance verification, slot search, or hold is performed.`,
    missing_info: missingForReferral(intake),
  });
}

async function handleOutOfAgeRange(
  item: InboxItem,
  router: RouterInput,
  classification: Classification,
  childAge: number,
): Promise<ItemOutput> {
  const intake = router.intake;
  const language = router.language;
  const greeting = greet(language, router.parent_first_name);
  const childRef = router.child_first_name || intake.child_name || (language === "es" ? "su familiar" : "your family member");

  await lookup_policy({ topic: "service_lines" });
  await ackReferringProvider(
    item,
    router,
    `patient is ${childAge}, outside our 0-${SERVICE_LINE_MAX_AGE} service line; we will not be proceeding with intake`,
  );

  const taskResult = await create_task({
    assignee: "front_desk",
    title: `Out-of-age-range referral: ${intake.child_name || "patient"} (age ${childAge})`,
    due: nextBusinessDay(item.received_at),
    notes: `Referral is for a ${childAge}-year-old; Cedar Kids serves 0-${SERVICE_LINE_MAX_AGE}. Front desk to send polite decline and, if possible, point the family toward an adult therapy resource.`,
  });

  const draft = await draft_message({
    recipient: contactPreference(item, intake),
    channel: replyChannel(item),
    body:
      language === "es"
        ? `${greeting} gracias por comunicarse con Cedar Kids Therapy. Atendemos exclusivamente a ninos y adolescentes de 0 a ${SERVICE_LINE_MAX_AGE} anos, asi que no podemos coordinar una evaluacion para ${childRef}. Un miembro de nuestro equipo le contactara con sugerencias de otros recursos en su area.`
        : `${greeting} thanks so much for reaching out to Cedar Kids Therapy. We specialize in children and adolescents ages 0-${SERVICE_LINE_MAX_AGE}, so we aren't able to schedule an evaluation for ${childRef}. Someone from our team will follow up with suggestions for adult therapy resources in the area.`,
    language,
  });

  return buildOutput({
    item,
    intake,
    classification,
    urgency: "P2",
    escalation: null,
    task_ids: [taskResult.data.task_id],
    draft_reply: draft.args.body as string,
    recommended_next_action:
      "Front desk to send polite decline and, if possible, refer the family to an adult therapy resource.",
    decision_rationale: `Cedar Kids service line is ages 0-${SERVICE_LINE_MAX_AGE} per policy; referral is for a ${childAge}-year-old. No insurance verification, slot search, or hold is performed because we cannot serve this patient.`,
    missing_info: missingForReferral(intake),
  });
}

async function neutralAcknowledgement(
  item: InboxItem,
  router: RouterInput,
): Promise<string> {
  const intake = router.intake;
  const language = router.language;
  const greeting = greet(language, router.parent_first_name);
  const body =
    language === "es"
      ? `${greeting} gracias por comunicarse con nosotros. Un miembro de nuestro equipo de Cedar Kids Therapy se pondra en contacto con usted directamente pronto.`
      : `${greeting} thank you for reaching out. Someone from our Cedar Kids Therapy team will be in touch with you directly soon.`;
  const draft = await draft_message({
    recipient: contactPreference(item, intake),
    channel: replyChannel(item),
    body,
    language: language === "es" ? "es" : "en",
  });
  return draft.args.body as string;
}

function greet(language: "en" | "es", firstName: string | null): string {
  if (language === "es") {
    return firstName ? `Hola ${firstName},` : "Hola,";
  }
  return firstName ? `Hi ${firstName},` : "Hi,";
}

function possessive(name: string): string {
  return name.endsWith("s") ? `${name}'` : `${name}'s`;
}

async function ackReferringProvider(
  item: InboxItem,
  router: RouterInput,
  status: string,
): Promise<void> {
  if (item.channel !== "fax_referral") return;

  const doctor = router.referring_doctor;
  const practice = router.referring_practice;
  if (!doctor && !practice) return;

  const childRef = router.intake.child_name || "the patient";
  const recipient = [doctor, practice].filter(Boolean).join(", ");
  const attachmentNote = item.attachments.length
    ? ` We received ${item.attachments.length} attachment${item.attachments.length === 1 ? "" : "s"}.`
    : "";

  await draft_message({
    recipient,
    channel: "phone",
    body: `Hi${doctor ? ` ${doctor}` : ""}, this is Cedar Kids Therapy. Confirming receipt of ${childRef}'s referral.${attachmentNote} Status: ${status}.`,
    language: "en",
  });
}

function referralOpener(
  item: InboxItem,
  language: "en" | "es",
  childRef: string,
): string {
  const fromPediatrician = item.channel === "fax_referral";
  if (language === "es") {
    return fromPediatrician
      ? `recibimos la referencia de ${childRef} de su pediatra.`
      : `muchas gracias por comunicarse con nosotros sobre ${childRef}.`;
  }
  return fromPediatrician
    ? `we received ${possessive(childRef)} referral from your pediatrician's office.`
    : `thanks so much for reaching out about ${childRef}.`;
}

interface BuildArgs {
  item: InboxItem;
  intake: ExtractedIntake;
  classification: Classification;
  urgency: Urgency;
  escalation: { reason: string; severity: "P0" | "P1" } | null;
  task_ids: string[];
  draft_reply: string | null;
  recommended_next_action: string;
  decision_rationale: string;
  missing_info: string[];
}

function buildOutput(args: BuildArgs): ItemOutput {
  return {
    item_id: args.item.id,
    classification: args.classification,
    urgency: args.urgency,
    requires_human_review: true,
    extracted_intake: args.intake,
    missing_info: args.missing_info,
    tools_called: getToolCallsForItem(args.item.id),
    recommended_next_action: args.recommended_next_action,
    draft_reply: args.draft_reply,
    task_ids: args.task_ids,
    escalation: args.escalation,
    decision_rationale: args.decision_rationale,
  };
}

function classify(item: InboxItem, intake: ExtractedIntake): Classification {
  const body = item.body.toLowerCase();

  if (hasSafeguardingSignal(body)) {
    return "safeguarding";
  }

  if (isReschedule(body) || isCancellation(body)) {
    return "scheduling";
  }

  if (isClinicalQuestion(body, intake)) {
    return "clinical_question";
  }

  if (isIncompleteReferral(item, intake)) {
    return "missing_paperwork";
  }

  if (item.channel === "fax_referral" || mentionsReferral(body)) {
    return "new_referral";
  }

  if (isExistingPatientRequest(body)) {
    return "existing_patient_request";
  }

  return "new_referral";
}

const SAFEGUARDING_TERMS = [
  "abuse",
  "abusive",
  "neglect",
  "hit ",
  "hits ",
  "hurt",
  "harm",
  "rough with",
  "rough w/",
  "unsafe",
  "scared of",
  "afraid of",
  "yells at",
  "screams at",
  "leaves alone",
  "leaving alone",
  "hungry all the time",
  "bruise",
];

function hasSafeguardingSignal(body: string): boolean {
  return SAFEGUARDING_TERMS.some((term) => body.includes(term));
}

function isReschedule(body: string): boolean {
  return /reschedul|move.*appointment|change.*appointment/.test(body);
}

function isCancellation(body: string): boolean {
  return /cancel|can'?t make|cant make|won'?t make|wont make/.test(body);
}

function isSameDayScheduling(body: string): boolean {
  return /today|same.day|this morning|this afternoon|tonight/i.test(body);
}

function isClinicalQuestion(body: string, intake: ExtractedIntake): boolean {
  if (intake.payer || intake.member_id) return false;
  return /is it normal|should i|should we|worried|advice|wait until|is this|are these/i.test(
    body,
  );
}

function isIncompleteReferral(item: InboxItem, intake: ExtractedIntake): boolean {
  if (item.channel !== "fax_referral") return false;
  if (/\[blank\]|\bblank\b|tbd|unknown|missing/i.test(item.body)) return true;
  const missing = missingForReferral(intake).length;
  return missing >= 3;
}

function mentionsReferral(body: string): boolean {
  return /referral|refer\b|eval|evaluation/i.test(body);
}

function isExistingPatientRequest(body: string): boolean {
  return /existing patient|my child'?s appointment|our appointment/i.test(body);
}

function extractIntake(item: InboxItem): ExtractedIntake {
  const body = item.body;
  return {
    child_name: extractChildName(item),
    dob_or_age: extractDob(body) || extractAge(body),
    parent_contact: extractParentContact(item),
    discipline: extractDisciplines(body),
    diagnosis_or_concern: extractConcern(body),
    payer: extractPayer(body),
    member_id: extractMemberId(body),
  };
}

function extractDob(body: string): string | null {
  const match = body.match(/\b(20\d{2}-\d{2}-\d{2})\b/);
  return match ? match[1] : null;
}

function extractAge(body: string): string | null {
  const yearOld = body.match(/(\d{1,2})[-\s]?year[-\s]?old/i);
  if (yearOld) return `age ${yearOld[1]}`;

  const heShe = body.match(/\b(?:he is|she is)\s+(\d{1,2})\b/i);
  if (heShe) return `age ${heShe[1]}`;

  const tiene = body.match(/tiene\s+(\d{1,2})\s+anos?/i);
  if (tiene) return `age ${tiene[1]}`;

  return null;
}

function extractChildName(item: InboxItem): string | null {
  const body = item.body;

  const explicit = body.match(/Child:\s*([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-'.]+)?)/);
  if (explicit) return explicit[1];

  const subject = item.subject;
  const subjMatch = subject.match(/(?:Referral|Voicemail|para|for|Referral for)[:\s]+([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)+)/);
  if (subjMatch) return subjMatch[1];

  const sonOrDaughter = body.match(/(?:my|our|mi|mis)\s+(?:son|daughter|child|hijo|hija)\s+([A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)?)/i);
  if (sonOrDaughter) return sonOrDaughter[1];

  const yearOld = body.match(/\d{1,2}[-\s]?year[-\s]?old\s+([A-Z][a-zA-Z\-']+)/);
  if (yearOld) return yearOld[1];

  const aboutChild = body.match(/(?:for|about)\s+([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+),?\s+(?:DOB|dob|age|is|threw|fell|missed)/);
  if (aboutChild) return aboutChild[1];

  const firstCapPair = body.match(/\b([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+)\s+(?:threw|fell|can'?t|cannot|missed|has been)/);
  if (firstCapPair) return firstCapPair[1];

  return null;
}

function extractParentContact(item: InboxItem): string | null {
  const body = item.body;
  const parts: string[] = [];

  const parentName = extractParentName(body, item.sender);
  if (parentName) parts.push(parentName);

  const phone = body.match(/(\d{3}-\d{3}-\d{4}|\(\d{3}\)\s*\d{3}-?\d{4}|555-\d{4}|\d{3}-\d{4})/);
  if (phone && !parts.join(", ").includes(phone[1])) parts.push(phone[1]);

  const email = body.match(/([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/);
  if (email && !parts.join(", ").includes(email[1])) parts.push(email[1]);

  if (parts.length === 0) {
    const senderEmail = item.sender.match(/<([^>]+)>/);
    if (senderEmail) return senderEmail[1];
    return null;
  }

  return parts.join(", ");
}

function extractParentName(body: string, sender: string): string | null {
  const explicit = body.match(/Parent(?:\/guardian)?:\s*([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+)/);
  if (explicit) return explicit[1];

  const iam = body.match(/I am (?:his|her|their)?\s*parent,?\s*([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+)/);
  if (iam) return iam[1];

  const thisIs = body.match(/this is\s+([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+)/i);
  if (thisIs) return thisIs[1];

  const soy = body.match(/soy\s+([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+)/i);
  if (soy) return soy[1];

  if (/fax|pediatric|practice|clinic|hospital|office/i.test(sender)) return null;

  const senderName = sender.match(/^([A-Z][a-zA-Z\-']+\s+[A-Z][a-zA-Z\-']+)/);
  if (senderName) return senderName[1];

  return null;
}

function extractDisciplines(body: string): Discipline[] | null {
  const found = new Set<Discipline>();
  if (/\bSLP\b|speech|articulation|habla|language patholog/i.test(body)) found.add("SLP");
  if (/\bOT\b|occupational|sensory|feeding/i.test(body)) found.add("OT");
  if (/\bPT\b|physical therap|toe walking|gait|gross motor/i.test(body)) found.add("PT");
  return found.size === 0 ? null : [...found];
}

function extractConcern(body: string): string | null {
  const explicit = body.match(/(?:Concern|Diagnosis\/concern|Concern\/Diagnosis):\s*([^.\n]+)/i);
  if (explicit) return explicit[1].trim();
  return null;
}

function extractPayer(body: string): string | null {
  const explicit = body.match(/Insurance:\s*([^.\n,]+)/i);
  if (explicit) {
    const value = explicit[1].trim();
    if (/\[blank\]|blank|none/i.test(value)) return null;
    return value;
  }
  const lower = body.toLowerCase();
  if (lower.includes("medicaid")) return "Medicaid";
  if (lower.includes("aetna")) return "Aetna";
  if (lower.includes("blue cross") || lower.includes("bcbs")) return "Blue Cross Blue Shield";
  if (lower.includes("united") || lower.includes("uhc")) return "UnitedHealthcare";
  if (lower.includes("kaiser")) return "Kaiser";
  if (lower.includes("cigna")) return "Cigna";
  return null;
}

function extractMemberId(body: string): string | null {
  const explicit = body.match(/Member ID:\s*([A-Za-z0-9\-]+)/i);
  if (explicit && !/blank/i.test(explicit[1])) return explicit[1];
  const inline = body.match(/\b(?:miembro|member)\s+([A-Z]{2,4}-?\d{3,7})/i);
  if (inline) return inline[1];
  const fallback = body.match(/\b([A-Z]{2,4}-\d{3,7})\b/);
  return fallback ? fallback[1] : null;
}

function extractPreferences(body: string): string | undefined {
  const match = body.match(/(?:Preferred availability|prefers?|prefer)[:\s]+([^.\n]+)/i);
  return match ? match[1].trim() : undefined;
}

function missingForReferral(intake: ExtractedIntake): string[] {
  const missing: string[] = [];
  if (!intake.child_name) missing.push("child's name");
  if (!intake.dob_or_age) missing.push("date of birth");
  if (!intake.parent_contact) missing.push("parent contact info");
  if (!intake.discipline) missing.push("requested service line");
  if (!intake.payer) missing.push("insurance payer");
  if (!intake.member_id) missing.push("insurance member ID");
  return missing;
}

function looksLikeDob(value: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(value);
}

function nameOverlaps(a: string, b: string): boolean {
  const tokens = (s: string) =>
    s
      .toLowerCase()
      .split(/[\s,]+/)
      .filter((t) => t.length >= 3);
  const at = tokens(a);
  const bt = tokens(b);
  return at.some((t) => bt.includes(t));
}

function contactPreference(item: InboxItem, intake: ExtractedIntake): string {
  const channel = replyChannel(item);
  const senderEmail = item.sender.match(/<([^>]+)>/)?.[1];
  const bodyEmail = item.body.match(/([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/)?.[1];
  const bodyPhone = item.body.match(/\d{3}-\d{3}-\d{4}|555-\d{4}/)?.[0];

  if (channel === "email") {
    return bodyEmail || senderEmail || intake.parent_contact || item.sender;
  }
  if (channel === "phone") {
    return bodyPhone || intake.parent_contact || item.sender;
  }
  if (channel === "portal") {
    return item.sender;
  }
  return item.sender;
}

function replyChannel(item: InboxItem): "portal" | "email" | "phone" {
  switch (item.channel) {
    case "email":
      return "email";
    case "portal_message":
      return "portal";
    case "voicemail_transcript":
      return "phone";
    case "fax_referral":
      return "email";
  }
}

function detectLanguage(item: InboxItem): "en" | "es" {
  const text = `${item.subject} ${item.body}`.toLowerCase();
  if (/espanol|en espa|hola|gracias|hija|hijo|anos|mensaje de voz/.test(text)) {
    return "es";
  }
  return "en";
}

function todayDate(receivedAt: string): string {
  return receivedAt.slice(0, 10);
}

function nextBusinessDay(receivedAt: string): string {
  const date = new Date(receivedAt);
  date.setDate(date.getDate() + 1);
  return date.toISOString().slice(0, 10);
}

function sameHourDue(receivedAt: string): string {
  const date = new Date(receivedAt);
  date.setHours(date.getHours() + 1);
  return date.toISOString();
}

function parseAge(dobOrAge: string | null, receivedAt: string): number | null {
  if (!dobOrAge) return null;
  if (/^\d{4}-\d{2}-\d{2}$/.test(dobOrAge)) {
    const dob = new Date(dobOrAge);
    const received = new Date(receivedAt);
    const years = (received.getTime() - dob.getTime()) / (365.25 * 24 * 60 * 60 * 1000);
    return Math.floor(years);
  }
  const match = dobOrAge.match(/(\d{1,2})/);
  return match ? Number.parseInt(match[1], 10) : null;
}

let cachedProviders: Provider[] | null = null;
function loadProviders(): Provider[] {
  if (cachedProviders !== null) return cachedProviders;
  const path = resolve(process.cwd(), "data/providers.json");
  if (!existsSync(path)) {
    cachedProviders = [];
    return cachedProviders;
  }
  cachedProviders = JSON.parse(readFileSync(path, "utf8")) as Provider[];
  return cachedProviders;
}

function providerCoversAge(providerId: string, childAge: number): boolean {
  const provider = loadProviders().find((p) => p.provider_id === providerId);
  if (!provider) return true;
  const match = provider.age_range.match(/^(\d+)-(\d+)$/);
  if (!match) return true;
  const min = Number.parseInt(match[1], 10);
  const max = Number.parseInt(match[2], 10);
  return childAge >= min && childAge <= max;
}

// Silence unused-import warning during refactors; these helpers are part of
// the agent's tool surface and may be wired into additional flows.
void hold_slot;
