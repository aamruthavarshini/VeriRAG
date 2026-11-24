
import re
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# CLAIM EXTRACTION
# --------------------------------------------------

ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.",
    "No.", "Fig.", "Rs.", "U.S.", "e.g.", "i.e."
]


def split_compound_claim(sentence):
    """
    Split simple compound factual statements.

    Example:
    TechNova Solutions was founded in 2018 and has
    300 employees.

    Becomes:
    TechNova Solutions was founded in 2018.
    TechNova Solutions has 300 employees.
    """

    pattern = re.compile(
        r"\s+and\s+(?="
        r"(?:is|are|was|were|has|have|had|"
        r"employs|owns|operates|offers|produces|"
        r"provides|develops|maintains|serves|"
        r"generated|earned|reported)\b)",
        re.IGNORECASE
    )

    parts = pattern.split(sentence.strip())

    if len(parts) == 1:
        return [sentence.strip()]

    first_part = parts[0].strip()

    # Identify the subject from the first clause.
    subject_match = re.match(
        r"^(.+?)\s+(?="
        r"is\b|are\b|was\b|were\b|has\b|have\b|had\b|"
        r"employs\b|owns\b|operates\b|offers\b|produces\b|"
        r"provides\b|develops\b|maintains\b|serves\b|"
        r"generated\b|earned\b|reported\b)",
        first_part,
        re.IGNORECASE
    )

    subject = subject_match.group(1) if subject_match else ""

    results = [first_part]

    for part in parts[1:]:
        part = part.strip()

        if not part:
            continue

        # Restore the subject if the second clause omits it.
        if subject and not re.match(
            r"^(?:the company|it|they|the organisation|"
            r"the organization)\b",
            part,
            re.IGNORECASE
        ):
            part = subject + " " + part

        results.append(part)

    return results


def extract_claims(answer):

    if not answer or not answer.strip():
        return []

    protected = answer

    # Protect abbreviations from sentence splitting.
    for abbreviation in ABBREVIATIONS:
        protected = protected.replace(
            abbreviation,
            abbreviation.replace(".", "<DOT>")
        )

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        protected.strip()
    )

    claims = []

    for sentence in sentences:

        sentence = sentence.replace("<DOT>", ".")
        sentence = sentence.strip(" \t-•")

        if len(sentence) < 12:
            continue

        if sentence.endswith(":") and len(sentence.split()) < 12:
            continue

        compound_claims = split_compound_claim(sentence)

        for claim in compound_claims:

            claim = claim.strip()

            if len(claim) >= 12:
                claims.append(claim)

    return claims


# --------------------------------------------------
# NUMERICAL CONSISTENCY
# --------------------------------------------------

def numeric_values(text):

    pattern = (
        r"(?:₹|Rs\.?|INR|\$|€|£)?\s*"
        r"\d[\d,]*(?:\.\d+)?\s*"
        r"(?:%|crore|cr|lakh|million|billion)?"
    )

    return [
        value.strip()
        for value in re.findall(pattern, text, re.I)
        if re.search(r"\d", value)
    ]


def numeric_consistency(claim, evidence):

    claim_numbers = [
        x.replace(" ", "").lower()
        for x in numeric_values(claim)
    ]

    evidence_numbers = [
        x.replace(" ", "").lower()
        for x in numeric_values(evidence)
    ]

    if not claim_numbers:
        return "No numeric values detected"

    if not evidence_numbers:
        return "Evidence contains no numeric values"

    missing = [
        x for x in claim_numbers
        if x not in evidence_numbers
    ]

    if missing:
        return (
            "Potential numeric mismatch: "
            + ", ".join(missing)
        )

    return "Numeric values match"


# --------------------------------------------------
# CLAIM VERIFIER
# --------------------------------------------------

class ClaimVerifier:

    def __init__(
        self,
        retriever,
        ollama,
        top_k=4,
        use_hf=False,
        hf_model="cross-encoder/nli-deberta-v3-small"
    ):

        self.retriever = retriever
        self.ollama = ollama
        self.top_k = top_k
        self.use_hf = use_hf
        self.hf_model_name = hf_model
        self.hf = None

    # --------------------------------------------------
    # OPTIONAL HUGGING FACE MODEL
    # --------------------------------------------------

    def load_hf(self):

        if self.hf is None:

            import torch

            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.hf_model_name
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                self.hf_model_name
            )

            model.eval()

            self.hf = (tokenizer, model, torch)

        return self.hf

    # --------------------------------------------------
    # CLASSIFY CLAIM AGAINST EVIDENCE
    # --------------------------------------------------

    def classify_pair(self, claim, evidence):

        if self.use_hf:

            tokenizer, model, torch = self.load_hf()

            inputs = tokenizer(
                evidence,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():

                logits = model(**inputs).logits[0]

                probabilities = torch.softmax(
                    logits,
                    dim=-1
                ).cpu().numpy()

            labels = model.config.id2label

            scores = {}

            for index, probability in enumerate(probabilities):

                label = str(
                    labels.get(index, "")
                ).lower()

                if "entail" in label:
                    scores["entailment"] = float(probability)

                elif "contrad" in label:
                    scores["contradiction"] = float(probability)

                elif "neutral" in label:
                    scores["neutral"] = float(probability)

            if len(scores) != 3:

                scores = {
                    "contradiction": float(probabilities[0]),
                    "entailment": float(probabilities[1]),
                    "neutral": float(probabilities[2])
                }

        else:

            prompt = f"""
You are a careful evidence-based claim verification system.

Determine the relationship between the evidence and claim.

CLAIM:
{claim}

EVIDENCE:
{evidence}

Return ONLY valid JSON:

{{
    "label": "entailment",
    "confidence": 0.95
}}

Allowed labels:

entailment = Evidence directly supports or confirms the claim.

contradiction = Evidence explicitly states something
incompatible with the claim.

neutral = Evidence does not establish the claim.

IMPORTANT RULES:

1. Missing information is NOT a contradiction.
2. Do not assume facts that are not in the evidence.
3. A different numerical value for the same property
   is a contradiction.
4. If the evidence discusses a different subject or property,
   do not treat that as a contradiction.
5. Verify only the specific claim provided.

Examples:

Claim: The company has offices across Europe.
Evidence: The company is headquartered in Bengaluru.
Label: neutral

Claim: The company has 300 employees.
Evidence: The company has 250 employees.
Label: contradiction

Claim: The company was founded in 2018.
Evidence: The company was founded in 2018.
Label: entailment
"""

            response = self.ollama.chat(
                prompt,
                system=(
                    "You are a cautious NLI classification system. "
                    "Return valid JSON only."
                )
            )

            match = re.search(
                r"\{.*\}",
                response,
                re.S
            )

            if not match:
                raise ValueError("Invalid NLI response")

            data = json.loads(match.group())

            label = data.get(
                "label",
                "neutral"
            ).lower()

            if label not in [
                "entailment",
                "contradiction",
                "neutral"
            ]:
                label = "neutral"

            confidence = float(
                data.get("confidence", 0.5)
            )

            confidence = max(
                0.0,
                min(1.0, confidence)
            )

            scores = {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 0.0
            }

            scores[label] = confidence

        label = max(
            scores,
            key=scores.get
        )

        status = {
            "entailment": "Supported",
            "contradiction": "Contradicted",
            "neutral": "Unsupported"
        }[label]

        return {
            "status": status,
            "nli_label": label,
            "confidence": scores[label]
        }

    # --------------------------------------------------
    # VERIFY COMPLETE ANSWER
    # --------------------------------------------------

    def verify_answer(self, answer):

        claims = extract_claims(answer)

        citation_names = re.findall(
            r"\[Source:\s*([^|\]]+)",
            answer,
            re.I
        )

        cited_sources = [
            name.strip()
            for name in citation_names
        ]

        results = []

        for claim in claims:

            evidence_items = self.retriever.search(
                claim,
                k=self.top_k,
                preferred_sources=cited_sources
            )

            if not evidence_items:
                continue

            pair_results = []

            for evidence in evidence_items:

                verdict = self.classify_pair(
                    claim,
                    evidence["text"]
                )

                pair_results.append(
                    (evidence, verdict)
                )

            priority = {
                "Supported": 2,
                "Contradicted": 2,
                "Unsupported": 1
            }

            pair_results.sort(
                key=lambda x: (
                    priority[x[1]["status"]],
                    x[1]["confidence"],
                    x[0]["similarity"]
                ),
                reverse=True
            )

            best_evidence, verdict = pair_results[0]

            numeric_check = numeric_consistency(
                claim,
                best_evidence["text"]
            )

            explanations = {
                "Supported":
                    "The selected evidence appears to support this claim.",

                "Contradicted":
                    "The selected evidence conflicts with this claim.",

                "Unsupported":
                    "The available evidence does not establish this claim."
            }

            explanation = explanations[
                verdict["status"]
            ]

            if "Potential numeric mismatch" in numeric_check:

                explanation += (
                    " A numerical inconsistency was also detected."
                )

            results.append({

                "claim": claim,

                "status": verdict["status"],

                "nli_label": verdict["nli_label"],

                "confidence": verdict["confidence"],

                "similarity": best_evidence["similarity"],

                "numeric_check": numeric_check,

                "citation_status": (
                    "Citation present"
                    if cited_sources
                    else "No explicit citation found"
                ),

                "explanation": explanation,

                "evidence": [
                    {
                        "source": ev["source"],
                        "chunk_id": ev["chunk_id"],
                        "text": ev["text"]
                    }
                    for ev, _ in pair_results[:3]
                ]

            })

        return results
