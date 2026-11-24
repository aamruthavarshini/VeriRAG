import unittest

from veritrag.verifier import (
    ClaimVerifier,
    extract_claims,
    numeric_consistency,
)
from veritrag.retrieval import Retriever
from veritrag.metrics import calculate_metrics, results_to_dataframe


class FakeOllama:
    def __init__(self, response):
        self.response = response

    def chat(self, prompt, system=None):
        return self.response


class FakeRetriever:
    def __init__(self, items):
        self.items = items

    def search(self, query, k=4):
        return self.items[:k]


class VeriRAGTests(unittest.TestCase):
    def test_zero_confidence_preserves_ollama_label(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":"contradiction","confidence":0.0}'
        ))
        result = verifier.classify_pair("A has 300 staff.", "A has 250 staff.")
        self.assertEqual(result["status"], "Contradicted")
        self.assertEqual(result["confidence"], 0.0)

    def test_json_with_surrounding_text_is_parsed(self):
        verifier = ClaimVerifier(None, FakeOllama(
            'Result:\n```json\n{"label":" Supported ","confidence":0.8}\n```\nDone'
        ))
        result = verifier.classify_pair("A was founded in 2018.", "A was founded in 2018.")
        self.assertEqual(result["status"], "Supported")

    def test_invalid_confidence_is_rejected(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":"entailment","confidence":"high"}'
        ))
        with self.assertRaises(ValueError):
            verifier.classify_pair("A has staff.", "A has staff.")

    def test_confidence_out_of_range_is_rejected(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":"entailment","confidence":95}'
        ))
        with self.assertRaises(ValueError):
            verifier.classify_pair("A has staff.", "A has staff.")

    def test_null_label_is_rejected(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":null,"confidence":0.8}'
        ))
        with self.assertRaises(ValueError):
            verifier.classify_pair("A has staff.", "A has staff.")

    def test_compound_claims_split_and_restore_subject(self):
        claims = extract_claims(
            "TechNova Solutions was founded in 2018 and has 300 employees."
        )
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[1], "TechNova Solutions has 300 employees.")

    def test_two_distinct_subjects_are_not_merged(self):
        claims = extract_claims(
            "Revenue was Rs. 35 crore and profit was Rs. 5 crore."
        )
        self.assertEqual(len(claims), 2)
        self.assertIn("profit was Rs. 5 crore", claims[1])


    def test_founded_and_ceo_compound_claims_split(self):
        claims = extract_claims("Alice founded the firm and is its CEO.")
        self.assertEqual(len(claims), 2)
        self.assertEqual(claims[1], "Alice is its CEO.")

    def test_relative_clause_is_not_mangled(self):
        sentence = "The company that was founded in 2018 has 250 employees and is based in Pune."
        claims = extract_claims(sentence)
        self.assertEqual(claims, [sentence])


    def test_bullet_claims_remain_separate(self):
        claims = extract_claims("- The company has 250 employees\n- Revenue was 35 crore")
        self.assertEqual(len(claims), 2)

    def test_short_factual_sentence_is_kept(self):
        self.assertIn("Costs fell.", extract_claims("Costs fell."))

    def test_citation_tag_is_not_a_claim(self):
        claims = extract_claims(
            "The company has 250 employees. [Source: facts.txt | Passage: 1]"
        )
        self.assertEqual(claims, ["The company has 250 employees."])

    def test_abbreviation_rs_does_not_split_sentence(self):
        claims = extract_claims("Revenue was Rs. 35 crore in 2024.")
        self.assertEqual(len(claims), 1)


    def test_numeric_same_property_conflict_overrides_neutral_nli(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":"neutral","confidence":0.0}'
        ))
        result = verifier.classify_pair(
            "TechNova has 300 employees.",
            "TechNova has 250 employees."
        )
        self.assertEqual(result["status"], "Contradicted")
        self.assertEqual(result["nli_label"], "contradiction")
        self.assertEqual(result["decision_method"], "numeric_property_rule:employee_count")

    def test_employee_conflict_is_detected_inside_multi_fact_passage(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":"neutral","confidence":0.0}'
        ))
        evidence = (
            "Company: TechNova Solutions. Founded in 2018. Revenue in 2024 "
            "was Rs. 35 crore. The company has 250 employees."
        )
        result = verifier.classify_pair(
            "TechNova Solutions has 300 employees.", evidence
        )
        self.assertEqual(result["status"], "Contradicted")
        self.assertEqual(result["decision_method"], "numeric_property_rule:employee_count")

    def test_numeric_difference_in_unrelated_properties_is_not_overridden(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":"neutral","confidence":0.7}'
        ))
        result = verifier.classify_pair(
            "TechNova has 300 employees.",
            "TechNova reported revenue of 250 crore."
        )
        self.assertEqual(result["status"], "Unsupported")
        self.assertEqual(result["decision_method"], "NLI")

    def test_revenue_numeric_conflict_is_detected(self):
        verifier = ClaimVerifier(None, FakeOllama(
            '{"label":"neutral","confidence":0.2}'
        ))
        result = verifier.classify_pair(
            "The company generated revenue of 50 crore.",
            "The company reported annual revenue of 35 crore."
        )
        self.assertEqual(result["status"], "Contradicted")
        self.assertEqual(result["decision_method"], "numeric_property_rule:revenue")

    def test_numeric_diagnostic_is_explicitly_preliminary(self):
        self.assertIn(
            "Potential numeric mismatch",
            numeric_consistency("Revenue was Rs. 50 crore.", "Revenue was Rs. 35 crore.")
        )

    def test_retriever_returns_no_unrelated_zero_score_passages(self):
        retriever = Retriever([
            {"source": "a.txt", "chunk_id": 1, "text": "apples and oranges"},
            {"source": "b.txt", "chunk_id": 1, "text": "cars and trucks"},
        ])
        self.assertEqual(retriever.search("quantum physics", k=2), [])

    def test_retriever_does_not_prefer_cited_source_over_relevance(self):
        retriever = Retriever([
            {"source": "cited.txt", "chunk_id": 1, "text": "unrelated gardening facts"},
            {"source": "correct.txt", "chunk_id": 1, "text": "TechNova has 250 employees"},
        ])
        results = retriever.search(
            "How many employees does TechNova have?",
            k=2,
            preferred_sources=["cited.txt"]
        )
        self.assertEqual(results[0]["source"], "correct.txt")

    def test_metrics_include_macro_f1_and_reject_unknown_labels(self):
        result = calculate_metrics(
            ["Supported", "Contradicted", "Unsupported"],
            ["Supported", "Contradicted", "Unsupported"]
        )
        self.assertEqual(result["macro_f1"], 1.0)
        with self.assertRaises(ValueError):
            calculate_metrics(["supported"], ["Supported"])

    def test_results_dataframe_handles_verification_errors(self):
        df = results_to_dataframe([{
            "claim": "Costs fell.",
            "status": "Verification Error",
            "explanation": "Mock failure"
        }])
        self.assertEqual(df.iloc[0]["Status"], "Verification Error")


    def test_citation_status_requires_source_and_passage_match(self):
        answer = "TechNova has 250 employees. [Source: facts.txt | Passage: 1]"
        items = [{"source": "facts.txt", "chunk_id": 1, "text": "TechNova has 250 employees."}]
        status = ClaimVerifier._citation_status_for_claim(
            answer, "TechNova has 250 employees.", items
        )
        self.assertEqual(status, "Citation matches retrieved evidence")

    def test_citation_with_txt_filename_period_matches(self):
        answer = "TechNova Solutions was founded in 2018. [Source: sample_company.txt | Passage: 1]"
        items = [{
            "source": "sample_company.txt",
            "chunk_id": 1,
            "text": "TechNova Solutions was founded in 2018.",
            "similarity": 0.8
        }]
        status = ClaimVerifier._citation_status_for_claim(
            answer, "TechNova Solutions was founded in 2018.", items
        )
        self.assertEqual(status, "Citation matches retrieved evidence")

    def test_no_retrieved_evidence_is_unsupported_not_error(self):
        verifier = ClaimVerifier(FakeRetriever([]), FakeOllama("{}"))
        result = verifier.verify_answer("The company has 250 employees.")[0]
        self.assertEqual(result["status"], "Unsupported")
        self.assertIsNone(result["error"])

    def test_verify_answer_uses_numeric_property_rule_for_employee_count(self):
        verifier = ClaimVerifier(
            FakeRetriever([{
                "source": "facts.txt",
                "chunk_id": 1,
                "text": "TechNova has 250 employees.",
                "similarity": 0.8
            }]),
            FakeOllama('{"label":"neutral","confidence":0.0}')
        )
        result = verifier.verify_answer(
            "TechNova has 300 employees. [Source: facts.txt | Passage: 1]"
        )[0]
        self.assertEqual(result["status"], "Contradicted")
        self.assertIn("numeric_property_rule", result["decision_method"])
        self.assertEqual(result["citation_status"], "Citation matches retrieved evidence")

    def test_verify_answer_reports_error_instead_of_crashing(self):
        verifier = ClaimVerifier(
            FakeRetriever([{
                "source": "facts.txt",
                "chunk_id": 1,
                "text": "TechNova has 250 employees.",
                "similarity": 0.7
            }]),
            FakeOllama("not json")
        )
        results = verifier.verify_answer("TechNova has 300 employees.")
        self.assertEqual(results[0]["status"], "Verification Error")
        self.assertEqual(results[0]["nli_label"], "error")


if __name__ == "__main__":
    unittest.main()
