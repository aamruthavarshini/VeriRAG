import json
import math
import re


# --------------------------------------------------
# CLAIM EXTRACTION
# --------------------------------------------------

ABBREVIATIONS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Fig.",
    "Rs.", "U.S.", "e.g.", "i.e."
)

PREDICATES = (
    "is", "are", "was", "were", "has", "have", "had",
    "employs", "owns", "operates", "offers", "produces",
    "provides", "develops", "maintains", "serves",
    "generated", "earned", "reported", "founded", "launched",
    "opened", "closed", "acquired", "released", "became",
    "increased", "decreased", "reduced", "expanded"
)


def _protect_abbreviations(text):
    for abbreviation in ABBREVIATIONS:
        text = re.sub(
            re.escape(abbreviation),
            abbreviation.replace(".", "<DOT>"),
            text,
            flags=re.IGNORECASE
        )
    return text


def split_compound_claim(sentence):
    """Split straightforward coordinated factual clauses without inventing subjects."""
    sentence = sentence.strip()
    predicate_group = "(?:" + "|".join(PREDICATES) + ")"
    pattern = re.compile(
        r"\s+and\s+(?=" + predicate_group +
        r"\b|[A-Za-z][\w'-]*\s+" + predicate_group + r"\b)",
        re.IGNORECASE
    )
    parts = pattern.split(sentence)
    if len(parts) == 1:
        return [sentence]

    first = parts[0].strip()
    predicate_pattern = re.compile(
        r"\s+(?=" + predicate_group + r"\b)",
        re.IGNORECASE
    )
    match = predicate_pattern.search(first)
    if not match:
        return [sentence]

    subject = first[:match.start()].strip()
    if (
        not subject
        or len(subject.split()) > 12
        or re.search(r"\b(?:that|which|who)\b", subject, re.IGNORECASE)
    ):
        return [sentence]

    claims = [first]
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        if re.match(predicate_group + r"\b", part, re.IGNORECASE):
            part = f"{subject} {part}"
        claims.append(part)
    return claims


def extract_claims(answer):
    """Extract factual sentence candidates; citation tags are metadata, not claims."""
    if not answer or not answer.strip():
        return []

    text = re.sub(
        r"\[Source:\s*[^\]]+\]",
        "",
        answer,
        flags=re.IGNORECASE
    )
    text = _protect_abbreviations(text)

    # Treat bullets/newlines as boundaries, but join ordinary wrapped lines.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = []
    for line in lines:
        is_list_item = bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line))
        if normalized and not is_list_item:
            previous = normalized[-1]
            if not re.search(r"[.!?]$", previous):
                normalized[-1] = previous + " " + line
                continue
        normalized.append(re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", line))

    text = "\n".join(normalized)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    claims = []
    for sentence in sentences:
        sentence = sentence.replace("<DOT>", ".").strip(" \t-•")
        if sentence.endswith(":") and len(sentence.split()) < 12:
            continue
        for claim in split_compound_claim(sentence):
            claim = claim.strip()
            # Keep short but meaningful factual statements such as "Costs fell."
            if len(claim) >= 8 and len(claim.split()) >= 2:
                claims.append(claim)
    return claims


# --------------------------------------------------
# NUMERICAL CONSISTENCY (DIAGNOSTIC ONLY)
# --------------------------------------------------

def numeric_values(text):
    pattern = (
        r"(?:₹|Rs\.?|INR|\$|€|£)?\s*"
        r"\d[\d,]*(?:\.\d+)?\s*"
        r"(?:%|percent|crore|cr|lakh|million|billion)?"
    )
    return [
        value.strip()
        for value in re.findall(pattern, text, re.IGNORECASE)
        if re.search(r"\d", value)
    ]


def numeric_consistency(claim, evidence):
    claim_numbers = [x.replace(" ", "").lower() for x in numeric_values(claim)]
    evidence_numbers = [x.replace(" ", "").lower() for x in numeric_values(evidence)]

    if not claim_numbers:
        return "No numeric values detected"
    if not evidence_numbers:
        return "Evidence contains no numeric values"

    missing = [x for x in claim_numbers if x not in evidence_numbers]
    if missing:
        return "Potential numeric mismatch: " + ", ".join(missing)
    return "Numeric values match"


_NUMERIC_PROPERTY_PATTERNS = {
    "employee_count": r"\b(?:employees?|workforce|staff|personnel|headcount)\b",
    "revenue": r"\b(?:revenue|sales|turnover|income)\b",
    "profit_loss": r"\b(?:profit|loss)\b",
    "operating_cost": r"\b(?:costs?|expenses?|expenditure)\b",
    "launch_date": r"\b(?:launched?|launch|released?|release)\b",
    "founding_date": r"\b(?:founded?|established|incorporated)\b",
    "contract_expiry": r"\b(?:contract|agreement)\b.*\b(?:expires?|expiry|expiration|ends?)\b|\b(?:expires?|expiry|expiration)\b.*\b(?:contract|agreement)\b",
    "emissions": r"\b(?:emissions?|carbon)\b",
}


def _numeric_property_conflict(claim, evidence):
    """Return a property key only for a clear same-property numeric disagreement.

    This intentionally handles a narrow set of explicit properties. It does not
    infer that arbitrary different numbers in a passage contradict one another.
    """
    claim_properties = {
        key for key, pattern in _NUMERIC_PROPERTY_PATTERNS.items()
        if re.search(pattern, claim, re.IGNORECASE)
    }
    evidence_properties = {
        key for key, pattern in _NUMERIC_PROPERTY_PATTERNS.items()
        if re.search(pattern, evidence, re.IGNORECASE)
    }
    shared = claim_properties & evidence_properties
    if len(shared) != 1:
        return None

    property_key = next(iter(shared))

    # Extract the number attached to the relevant property, rather than
    # requiring the entire passage to contain only one number. Real retrieved
    # passages commonly contain several unrelated dates and quantities.
    number = r"(\d[\d,]*(?:\.\d+)?)"
    patterns = {
        "employee_count": [
            rf"{number}\s*(?:employees?|staff|personnel|people)",
            rf"(?:workforce|headcount)\s*(?:of|is|was|:)?\s*{number}",
        ],
        "revenue": [
            rf"(?:revenue|sales|turnover|income)[^.!?;]{{0,45}}?{number}(?:\s*(?:crore|cr|lakh|million|billion|%|percent))?",
            rf"{number}\s*(?:crore|cr|lakh|million|billion)\s+(?:in\s+)?(?:revenue|sales|turnover|income)",
        ],
        "profit_loss": [
            rf"(?:profit|loss)[^.!?;]{{0,35}}?{number}(?:\s*(?:crore|cr|lakh|million|billion|%|percent))?",
            rf"{number}\s*(?:crore|cr|lakh|million|billion)\s+(?:profit|loss)",
        ],
        "operating_cost": [
            rf"(?:costs?|expenses?|expenditure)[^.!?;]{{0,35}}?{number}(?:\s*(?:crore|cr|lakh|million|billion|%|percent))?",
            rf"{number}\s*(?:%|percent|crore|cr|lakh|million|billion)?\s+(?:costs?|expenses?|expenditure)",
        ],
        "launch_date": [
            rf"(?:launched?|launch|released?|release)[^.!?;]{{0,35}}?{number}",
        ],
        "founding_date": [
            rf"(?:founded?|established|incorporated)[^.!?;]{{0,35}}?{number}",
        ],
        "contract_expiry": [
            rf"(?:expires?|expiry|expiration|ends?)[^.!?;]{{0,35}}?{number}",
        ],
        "emissions": [
            rf"(?:emissions?|carbon)[^.!?;]{{0,35}}?{number}(?:\s*(?:%|percent|tonnes?|tons?|kg|kilograms?))?",
        ],
    }

    def property_value(text):
        for pattern in patterns.get(property_key, []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # The capture group is the numeric value bound to the property.
                return match.group(1).replace(",", "")
        return None

    claim_value = property_value(claim)
    evidence_value = property_value(evidence)
    if claim_value is not None and evidence_value is not None and claim_value != evidence_value:
        return property_key
    return None


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

    def load_hf(self):
        if self.hf is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_name)
            model.eval()
            self.hf = (tokenizer, model, torch)
        return self.hf

    @staticmethod
    def _parse_ollama_response(response):
        if not isinstance(response, str):
            raise ValueError("The verifier returned a non-text response.")

        decoder = json.JSONDecoder()
        start = response.find("{")
        if start < 0:
            raise ValueError("Verifier response did not contain a JSON object.")

        try:
            data, _ = decoder.raw_decode(response[start:])
        except json.JSONDecodeError as error:
            raise ValueError("Verifier returned invalid JSON.") from error

        if not isinstance(data, dict):
            raise ValueError("Verifier JSON must be an object.")

        raw_label = data.get("label")
        if not isinstance(raw_label, str):
            raise ValueError("Verifier JSON is missing a valid string label.")

        label = raw_label.strip().lower()
        aliases = {
            "entailment": "entailment",
            "entailed": "entailment",
            "supported": "entailment",
            "contradiction": "contradiction",
            "contradicted": "contradiction",
            "neutral": "neutral",
            "unsupported": "neutral"
        }
        if label not in aliases:
            raise ValueError(f"Unknown verifier label: {raw_label!r}")
        label = aliases[label]

        raw_confidence = data.get("confidence")
        if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
            raise ValueError("Verifier confidence must be a number from 0 to 1.")
        confidence = float(raw_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Verifier confidence must be between 0 and 1.")
        return label, confidence

    def _classify_hf(self, claim, evidence):
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
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

        id2label = model.config.id2label
        scores = {}
        for index, probability in enumerate(probabilities):
            label = str(id2label.get(index, id2label.get(str(index), ""))).strip().lower()
            if "entail" in label:
                scores["entailment"] = float(probability)
            elif "contrad" in label:
                scores["contradiction"] = float(probability)
            elif "neutral" in label:
                scores["neutral"] = float(probability)

        if set(scores) != {"entailment", "contradiction", "neutral"}:
            raise RuntimeError(
                "Could not identify all three NLI labels from the Hugging Face model "
                "configuration. Refusing to guess the model's label order."
            )
        return scores

    def classify_pair(self, claim, evidence):
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Claim must be a non-empty string.")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Evidence must be a non-empty string.")

        if self.use_hf:
            scores = self._classify_hf(claim, evidence)
            # Deterministic tie-break: do not silently prefer entailment.
            label = max(
                ("contradiction", "neutral", "entailment"),
                key=lambda item: scores[item]
            )
            confidence = scores[label]
        else:
            prompt = f"""
You are a careful evidence-based claim verification system.
Determine the relationship between the evidence and claim.

CLAIM:
{claim}

EVIDENCE:
{evidence}

Return ONLY one JSON object with:
{{"label":"entailment|contradiction|neutral","confidence":0.0}}

Definitions:
- entailment: evidence directly supports the complete claim.
- contradiction: evidence explicitly states an incompatible fact about the same subject and property.
- neutral: evidence does not establish the claim.

Rules:
1. Missing information is NOT a contradiction.
2. Do not assume facts absent from the evidence.
3. A different value for the same subject and property is a contradiction.
4. Different subjects or properties are not contradictions.
5. Use neutral when evidence is incomplete or ambiguous.
6. Confidence must be a number between 0 and 1.

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
                    "You are a cautious natural-language-inference classifier. "
                    "Return one valid JSON object only."
                )
            )
            label, confidence = self._parse_ollama_response(response)

        decision_method = "NLI"
        numeric_property = _numeric_property_conflict(claim, evidence)
        if numeric_property:
            # A narrowly scoped deterministic rule catches explicit numeric
            # disagreements for the same named property (e.g. 300 vs 250 employees).
            # It does not treat unrelated numbers as contradictions.
            label = "contradiction"
            decision_method = f"numeric_property_rule:{numeric_property}"

        status = {
            "entailment": "Supported",
            "contradiction": "Contradicted",
            "neutral": "Unsupported"
        }[label]
        return {
            "status": status,
            "nli_label": label,
            "confidence": float(confidence),
            "decision_method": decision_method
        }

    @staticmethod
    def _citation_status(claim, evidence_items):
        """Check whether this claim's citation points to one of its retrieved passages."""
        tags = re.findall(
            r"\[Source:\s*([^|\]]+)(?:\|\s*Passage:\s*(\d+))?\s*\]",
            claim,
            flags=re.IGNORECASE
        )
        if not tags:
            return "No claim-level citation marker"
        for source, passage in tags:
            for item in evidence_items:
                source_matches = source.strip().casefold() == str(item.get("source", "")).casefold()
                passage_matches = not passage or str(item.get("chunk_id")) == passage
                if source_matches and passage_matches:
                    return "Citation matches retrieved evidence"
        return "Citation does not match retrieved evidence"

    @staticmethod
    def _citation_status_for_claim(answer, claim, evidence_items):
        # Protect periods inside filenames such as sample_company.txt before
        # splitting sentences; otherwise the citation tag is broken in half.
        protected_answer = re.sub(
            r"\[Source:[^\]]+\]",
            lambda match: match.group(0).replace(".", "<CITATION_DOT>"),
            answer,
            flags=re.IGNORECASE
        )
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", protected_answer)
        sentences = []
        for sentence in raw_sentences:
            sentence = sentence.replace("<CITATION_DOT>", ".")
            if re.fullmatch(r"\s*\[Source:[^\]]+\]\s*", sentence, re.I) and sentences:
                sentences[-1] += " " + sentence
            else:
                sentences.append(sentence)

        claim_terms = set(re.findall(r"\w+", claim.casefold()))
        best_tags = []
        best_overlap = 0.0
        for sentence in sentences:
            tags = re.findall(
                r"\[Source:\s*([^|\]]+)(?:\|\s*Passage:\s*(\d+))?\s*\]",
                sentence,
                flags=re.IGNORECASE
            )
            if not tags:
                continue
            sentence_text = re.sub(r"\[Source:[^\]]+\]", "", sentence, flags=re.I)
            sentence_terms = set(re.findall(r"\w+", sentence_text.casefold()))
            overlap = len(claim_terms & sentence_terms) / max(1, len(claim_terms))
            if overlap > best_overlap:
                best_overlap = overlap
                best_tags = tags

        if not best_tags or best_overlap < 0.4:
            return "No claim-level citation marker"

        for source, passage in best_tags:
            for item in evidence_items:
                source_matches = source.strip().casefold() == str(item.get("source", "")).casefold()
                passage_matches = not passage or str(item.get("chunk_id")) == passage
                if source_matches and passage_matches:
                    return "Citation matches retrieved evidence"
        return "Citation does not match retrieved evidence"

    def verify_answer(self, answer):
        if self.retriever is None:
            raise RuntimeError("A document retriever is required to verify an answer.")

        claims = extract_claims(answer)
        results = []

        for claim in claims:
            # Capture citations from the original claim sentence before extraction
            # removes citation markers.
            sentence_citations = re.findall(
                r"\[Source:\s*([^|\]]+)(?:\|\s*Passage:\s*(\d+))?\s*\]",
                answer,
                flags=re.IGNORECASE
            )

            try:
                evidence_items = self.retriever.search(claim, k=self.top_k)
                if not evidence_items:
                    results.append(self._unsupported_result(
                        claim, "No sufficiently relevant passage was retrieved."
                    ))
                    continue

                pair_results = []
                errors = []
                for evidence in evidence_items:
                    try:
                        verdict = self.classify_pair(claim, evidence["text"])
                        pair_results.append((evidence, verdict))
                    except Exception as error:
                        errors.append(str(error))

                if not pair_results:
                    results.append(self._error_result(
                        claim,
                        "Verification failed for all retrieved passages: " +
                        (errors[0] if errors else "unknown error")
                    ))
                    continue

                # Choose the strongest non-neutral assessment; confidence and
                # lexical similarity break ties. A neutral result cannot outrank
                # a supported/contradicted assessment.
                priority = {"Supported": 2, "Contradicted": 2, "Unsupported": 1}
                pair_results.sort(
                    key=lambda pair: (
                        priority[pair[1]["status"]],
                        pair[1]["confidence"],
                        pair[0].get("similarity", 0.0)
                    ),
                    reverse=True
                )
                best_evidence, verdict = pair_results[0]
                numeric_check = numeric_consistency(claim, best_evidence["text"])
                explanation = {
                    "Supported": "The selected evidence appears to support this claim.",
                    "Contradicted": "The selected evidence conflicts with this claim.",
                    "Unsupported": "The available evidence does not establish this claim."
                }[verdict["status"]]

                if verdict.get("decision_method", "NLI").startswith("numeric_property_rule:"):
                    explanation += (
                        " A deterministic same-property numeric comparison overrode "
                        "the NLI label because the claim and evidence state different values."
                    )
                elif "Potential numeric mismatch" in numeric_check:
                    explanation += " A preliminary numerical inconsistency was also detected."

                # Citation association is checked against the answer sentence
                # containing the claim, rather than treating any answer citation
                # as proof that every claim is cited.
                claim_citation_status = self._citation_status_for_claim(
                    answer, claim, [item for item, _ in pair_results]
                )

                results.append({
                    "claim": claim,
                    "status": verdict["status"],
                    "nli_label": verdict["nli_label"],
                    "confidence": verdict["confidence"],
                    "decision_method": verdict.get("decision_method", "NLI"),
                    "similarity": float(best_evidence.get("similarity", 0.0)),
                    "numeric_check": numeric_check,
                    "citation_status": claim_citation_status,
                    "explanation": explanation,
                    "evidence": [
                        {
                            "source": item["source"],
                            "chunk_id": item["chunk_id"],
                            "text": item["text"]
                        }
                        for item, _ in pair_results[:3]
                    ],
                    "error": None
                })
            except Exception as error:
                results.append(self._error_result(claim, str(error)))

        return results

    @staticmethod
    def _unsupported_result(claim, message):
        return {
            "claim": claim,
            "status": "Unsupported",
            "nli_label": "neutral",
            "confidence": 0.0,
            "decision_method": "No evidence retrieved",
            "similarity": 0.0,
            "numeric_check": "Not evaluated",
            "citation_status": "Not evaluated",
            "explanation": message,
            "evidence": [],
            "error": None
        }

    @staticmethod
    def _error_result(claim, message):
        return {
            "claim": claim,
            "status": "Verification Error",
            "nli_label": "error",
            "confidence": 0.0,
            "decision_method": "Verification error",
            "similarity": 0.0,
            "numeric_check": "Not evaluated",
            "citation_status": "Not evaluated",
            "explanation": message,
            "evidence": [],
            "error": message
        }
