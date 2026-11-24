# Changelog

## Updated research prototype

- Fixed zero-confidence verdict handling so Ollama's returned label is preserved.
- Added strict JSON, label and confidence validation for Ollama NLI responses.
- Added explicit verification-error results instead of silently assigning a factual label when classification fails.
- Improved compound-claim splitting, abbreviation handling, short claims and bullet-list extraction.
- Added retrieval similarity filtering and removed citation-based source ranking boosts.
- Added safer text/PDF processing with readable per-file error reporting.
- Added claim-associated source/passage citation matching as a heuristic check.
- Added macro-F1, label validation and per-example evaluation export.
- Added 21 automated unit tests.
- Updated README to accurately describe TF-IDF retrieval, evaluation scope and prototype limitations.
- Removed unused `tabulate` dependency and expanded `.gitignore`.


## Final validation update

- Added a narrow deterministic same-property numeric conflict rule for explicitly comparable values (including employee counts and revenue).
- Exposed the final decision method and clarified that the displayed confidence is the underlying NLI model confidence.
- Added regression tests ensuring unrelated numbers do not trigger the numeric override.
- Added the 30-row balanced development CSV; documentation explicitly states that it is not a held-out benchmark.
- Hardened evaluation CSV read and empty-cell handling.

## Final validation correction (24 November 2026)
- Fixed numeric property comparison to extract the value associated with the relevant property from a multi-fact evidence passage, instead of requiring the passage to contain exactly one number.
- Fixed claim-level citation parsing so periods in filenames such as `.txt` do not break citation tags during sentence association.
- Added regression tests for a multi-fact company passage and `.txt` source citations.
- Validation: 27 automated tests passed; Python compilation passed. Live Ollama/Streamlit behaviour still requires local confirmation.
