# VeriRAG

## AI Answer Verification and Source Attribution System

VeriRAG is a local-first research prototype for retrieval-augmented question answering and claim-level assessment. It uses uploaded documents as evidence, a local Ollama model to generate answers and classify claims, and a TF-IDF retriever to locate passages.

## Features

- Extract text from PDF, TXT and Markdown uploads.
- Chunk documents and retain source filename and chunk number.
- Retrieve passages with TF-IDF and cosine similarity (lexical similarity, not semantic embeddings).
- Generate document-grounded answers with Ollama.
- Extract sentence-level and straightforward compound claims.
- Classify claim/evidence pairs as Supported, Contradicted or Unsupported.
- Show retrieved evidence and a preliminary numerical consistency diagnostic.
- Apply a narrow same-property numeric conflict rule for explicit comparisons such as employee count or revenue, and display when that rule overrides the NLI label.
- Check whether a claim's source marker matches a retrieved source/passage.
- Evaluate labelled claim/evidence pairs and export per-example results.

## Requirements

- Python 3.11 recommended
- Ollama installed locally
- `llama3.2:latest` pulled in Ollama
- Windows, macOS or Linux

The default configuration uses local Ollama. No paid API is required.

## Setup (Windows)

1. Install Python 3.11 and Ollama.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Download the model:

   ```powershell
   ollama pull llama3.2:latest
   ```

6. Start the application:

   ```powershell
   streamlit run app.py
   ```

Open the local address printed by Streamlit (normally http://localhost:8501).

Run the automated tests from the project directory:

```powershell
python -m unittest discover -s tests -v
```

You can also use `run.bat` after installing dependencies and pulling the model.

## Optional Hugging Face NLI backend

The Hugging Face backend is optional and is not installed by the default requirements file. To try it, install compatible `torch` and `transformers` packages separately. The first run may download model files, so it may not be suitable for an offline environment. VeriRAG refuses to guess label ordering if the selected model does not expose recognizable NLI labels.

## Evaluation

The Evaluation tab expects a CSV with these columns:

- `claim`
- `evidence`
- `gold_label`

Allowed gold labels are exactly `Supported`, `Contradicted`, and `Unsupported`.

The bundled `sample_evaluation.csv` is a five-row demonstration file. The `veritrag_evaluation_30_development.csv` file is a 30-row balanced development dataset (10 examples per class) supplied for local experiments. It was not independently collected as a held-out benchmark and must not be reported as an unbiased final test set. The evaluation tab classifies each supplied claim/evidence pair. It does **not** measure the entire RAG pipeline, claim extraction, document retrieval, or citation association. Report the dataset size and construction method alongside results. A small development dataset is not evidence of generalisation; use a separate, independently labelled held-out set for final project evaluation.

## Interpretation and limitations

- **Supported**: the selected evidence appears to support the claim.
- **Contradicted**: the selected evidence explicitly conflicts with the claim.
- **Unsupported**: the selected evidence does not establish the claim.
- **Verification Error**: the model response or processing failed; this is not a factual verdict.

This is a research prototype, not a guarantee of factual truth. The default retriever is lexical TF-IDF, not a semantic embedding model. Numeric checks are preliminary string-based diagnostics and do not normalize equivalent units (for example, lakh versus million). A separate deterministic rule only overrides the NLI label when both claim and evidence contain exactly one numeric value and share one explicitly recognized property category; it is deliberately narrow and is not a general numeric reasoning system. The displayed NLI confidence remains the model confidence, even when the deterministic rule determines the final status. Citation matching is a heuristic association between a claim sentence's marker and retrieved source/passage metadata; it does not prove that the source itself is reliable or that the cited text fully supports the claim. The NLI model's confidence is not calibrated probability.

Long documents, complex compound sentences, tables, scanned PDFs and ambiguous references may reduce performance. Review the displayed evidence rather than treating labels as definitive.

### Final validation notes
The deterministic numeric-property rule only overrides NLI when it can extract a numeric value attached to the same recognized property in both claim and evidence (for example, employee count). It is intentionally narrow and should not be treated as a general-purpose numeric reasoning engine. Claim-level citation parsing protects periods in filenames such as `.txt` when associating citation markers with claims.
