import type { EvidenceBundle, FieldQrCodeDecision } from "@/lib/api/client";

export function unsafeEvidenceSource(
  evidence: EvidenceBundle,
  sourceType: string,
  sourceId: string,
): boolean {
  return (evidence.security_findings ?? []).some((finding) => (
    finding.source_type === sourceType && finding.source_id === sourceId
  ));
}

export function isReviewableVideoEvent(eventType: string): boolean {
  return [
    "visual_candidate",
    "motion_candidate",
    "signal_pattern_candidate",
    "action_sequence_candidate",
  ].includes(eventType);
}

export function fieldRecognitionQrReviewComplete(
  evidence: EvidenceBundle,
  decisions: Record<string, FieldQrCodeDecision>,
): boolean {
  return (evidence.qr_codes ?? []).every((candidate) => {
    const decision = decisions[candidate.candidate_id];
    if (!decision) return false;
    if (candidate.security_findings.length > 0 && decision.disposition === "ACCEPTED") {
      return false;
    }
    return decision.disposition !== "CORRECTED" || Boolean(decision.corrected_text?.trim());
  });
}

