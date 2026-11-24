
# VeriRAG

## AI Answer Verification and Source Attribution System

VeriRAG is a local-first Retrieval-Augmented Generation application that verifies individual claims in AI-generated answers against uploaded source documents.

### Features

1. PDF and text document processing.
2. Document chunking.
3. TF-IDF-based evidence retrieval.
4. Local LLM answer generation using Ollama.
5. Sentence-level claim extraction.
6. NLI-based claim verification.
7. Semantic similarity scoring.
8. Numerical inconsistency detection.
9. Evidence traceability.
10. Evaluation using accuracy, precision, recall and F1.

### Technologies

- Python
- Streamlit
- Ollama
- Llama 3.2
- Scikit-learn
- PyMuPDF
- Pandas
- Hugging Face Transformers (optional)

### Installation

1. Install Python 3.11.
2. Install Ollama.
3. Download the model:

   ollama pull llama3.2:latest

4. Open the project directory.
5. Run run.bat.

### Application

Open the local Streamlit address shown in the terminal.

Usually:

http://localhost:8501

### Verification labels

Supported:
The evidence supports the claim.

Contradicted:
The evidence conflicts with the claim.

Unsupported:
The evidence does not establish the claim.

### Limitations

The prototype cannot independently establish whether a source document is truthful. Its classifications must be evaluated against manually labelled examples. Numerical checks are preliminary and do not normalize all equivalent units.
