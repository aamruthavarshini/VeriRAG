
import streamlit as st
import pandas as pd

from veritrag.document_processor import extract_documents
from veritrag.retrieval import Retriever
from veritrag.generator import OllamaClient
from veritrag.verifier import ClaimVerifier
from veritrag.metrics import (
    calculate_metrics,
    results_to_dataframe
)


st.set_page_config(
    page_title="VeriRAG",
    page_icon="🛡️",
    layout="wide"
)


st.title("🛡️ VeriRAG")

st.subheader(
    "AI Answer Verification and Source Attribution System"
)

st.caption(
    "Assess answer claims against retrieved passages, show evidence, and check whether "
    "claim-level source markers match retrieved source passages. This is a research prototype, "
    "not a guarantee of factual correctness."
)


if "documents" not in st.session_state:
    st.session_state.documents = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "results" not in st.session_state:
    st.session_state.results = None

if "answer" not in st.session_state:
    st.session_state.answer = ""


with st.sidebar:

    st.header("Configuration")

    ollama_url = st.text_input(
        "Ollama URL",
        "http://localhost:11434"
    )

    model = st.text_input(
        "Ollama model",
        "llama3.2:latest"
    )

    top_k = st.slider(
        "Evidence passages",
        1,
        8,
        4
    )

    use_hf = st.checkbox(
        "Use pretrained Hugging Face NLI model"
    )


ollama = OllamaClient(
    ollama_url,
    model
)


connected, model_info = ollama.check()

if connected:

    st.sidebar.success(
        "Ollama connected"
    )

else:

    st.sidebar.error(
        "Ollama disconnected"
    )


tab1, tab2, tab3 = st.tabs([
    "Documents and RAG",
    "Claim Verification",
    "Evaluation"
])


with tab1:

    st.header("Upload documents")

    uploads = st.file_uploader(
        "Upload PDF, TXT or MD documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button(
        "Process Documents",
        type="primary"
    ):

        if not uploads:

            st.warning(
                "Please upload documents first."
            )

        else:

            with st.spinner(
                "Processing documents..."
            ):

                docs = extract_documents(
                    uploads
                )

                for file_error in extract_documents.last_errors:
                    st.warning(file_error)

                if not docs:

                    st.error(
                        "No readable text found."
                    )

                else:

                    st.session_state.documents = docs

                    st.session_state.retriever = Retriever(
                        docs
                    )

                    st.success(
                        f"{len(docs)} passages indexed successfully."
                    )

    if st.session_state.documents:

        st.info(
            f"{len(st.session_state.documents)} passages available."
        )

    st.divider()

    st.header("Ask your documents")

    question = st.text_area(
        "Enter your question",
        placeholder="What was the company's annual revenue?"
    )

    if st.button(
        "Generate AI Answer",
        type="primary",
        disabled=not st.session_state.documents
    ):

        if not question.strip():

            st.warning(
                "Enter a question."
            )

        else:

            with st.spinner(
                "Generating answer..."
            ):

                passages = st.session_state.retriever.search(
                    question,
                    k=top_k
                )

                try:
                    answer = ollama.generate_answer(
                        question,
                        passages
                    )
                    st.session_state.answer = answer
                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")

    if st.session_state.answer:

        st.subheader("Generated Answer")

        st.write(
            st.session_state.answer
        )

        if st.button(
            "Verify this answer"
        ):

            st.session_state.verify_text = (
                st.session_state.answer
            )

            st.rerun()


with tab2:

    st.header("Claim-level verification")

    answer_text = st.text_area(
        "Paste an AI-generated answer",
        value=st.session_state.get(
            "verify_text",
            ""
        ),
        height=180
    )

    if st.button(
        "Verify Claims",
        type="primary",
        disabled=not st.session_state.documents
    ):

        if not answer_text.strip():

            st.warning(
                "Please enter an answer."
            )

        else:

            verifier = ClaimVerifier(
                retriever=st.session_state.retriever,
                ollama=ollama,
                top_k=top_k,
                use_hf=use_hf
            )

            with st.spinner(
                "Checking claims against evidence..."
            ):

                try:

                    st.session_state.results = (
                        verifier.verify_answer(
                            answer_text
                        )
                    )

                except Exception as error:

                    st.error(str(error))


    results = st.session_state.results

    if results is not None:

        df = results_to_dataframe(
            results
        )

        total = len(df)

        supported = (
            df["Status"] == "Supported"
        ).sum()

        contradicted = (
            df["Status"] == "Contradicted"
        ).sum()

        unsupported = (
            df["Status"] == "Unsupported"
        ).sum()

        st.divider()

        st.subheader("Verification Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Claims",
            total
        )

        c2.metric(
            "Supported",
            int(supported)
        )

        c3.metric(
            "Contradicted",
            int(contradicted)
        )

        c4.metric(
            "Unsupported",
            int(unsupported)
        )

        # Expose claim-level citation verification in both the table and details.
        # The factual verdict and citation verdict are separate: a claim can be
        # Supported while its supplied source/passage marker is incorrect.
        citation_statuses = [
            result.get("citation_status", "Not evaluated")
            for result in results
        ]
        df["Citation Status"] = citation_statuses

        citation_matches = sum(
            status == "Citation matches retrieved evidence"
            for status in citation_statuses
        )
        citation_mismatches = sum(
            status == "Citation does not match retrieved evidence"
            for status in citation_statuses
        )
        citation_missing = sum(
            status == "No claim-level citation marker"
            for status in citation_statuses
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        citation_col1, citation_col2, citation_col3 = st.columns(3)
        citation_col1.metric("Citations matched", citation_matches)
        citation_col2.metric("Citation mismatches", citation_mismatches)
        citation_col3.metric("Claims without citation", citation_missing)

        for index, result in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Claim {index}: {result.get('status', 'Unknown')}"
            ):

                st.write(
                    "**Claim:**",
                    result.get("claim", "")
                )

                st.write(
                    "**Explanation:**",
                    result.get("explanation", "")
                )

                st.write(
                    "**NLI confidence:**",
                    round(float(result.get("confidence", 0.0)), 3)
                )

                st.write(
                    "**Decision method:**",
                    result.get("decision_method", "NLI")
                )

                st.write(
                    "**Similarity:**",
                    round(float(result.get("similarity", 0.0)), 3)
                )

                st.write(
                    "**Numerical check:**",
                    result.get("numeric_check", "Not evaluated")
                )

                citation_status = result.get(
                    "citation_status",
                    "Not evaluated"
                )
                if citation_status == "Citation matches retrieved evidence":
                    st.success(f"Citation status: {citation_status}")
                elif citation_status == "Citation does not match retrieved evidence":
                    st.error(f"Citation status: {citation_status}")
                elif citation_status == "No claim-level citation marker":
                    st.warning(f"Citation status: {citation_status}")
                else:
                    st.info(f"Citation status: {citation_status}")

                if result.get("error"):
                    st.error(f"Verification error: {result['error']}")

                st.write("**Evidence:**")

                for evidence in result.get("evidence", []):
                    st.markdown(
                        f"**{evidence.get('source', 'Unknown source')} — "
                        f"Passage {evidence.get('chunk_id', 'Unknown')}**"
                    )
                    st.write(evidence.get("text", ""))

        st.download_button(
            "Download CSV Report",
            df.to_csv(index=False),
            "veritrag_report.csv",
            "text/csv"
        )


with tab3:

    st.header("Model Evaluation")

    st.write(
        "Upload a labelled evaluation CSV."
    )

    file = st.file_uploader(
        "Evaluation dataset",
        type=["csv"],
        key="evaluation_file"
    )

    if file:

        try:
            evaluation_data = pd.read_csv(file)
        except Exception as error:
            st.error(f"Could not read the evaluation CSV: {error}")
            evaluation_data = None

        required = {
            "claim",
            "evidence",
            "gold_label"
        }

        if evaluation_data is None:
            pass

        elif not required.issubset(
            evaluation_data.columns
        ):

            st.error(
                "CSV needs claim, evidence and gold_label columns."
            )

        elif st.button(
            "Run Evaluation"
        ):

            verifier = ClaimVerifier(
                retriever=None,
                ollama=ollama,
                use_hf=use_hf
            )

            predictions = []
            evaluation_rows = []

            with st.spinner("Evaluating examples..."):
                for row_index, row in evaluation_data.iterrows():
                    try:
                        if pd.isna(row["claim"]) or pd.isna(row["evidence"]):
                            raise ValueError("Claim and evidence must not be empty.")
                        result = verifier.classify_pair(
                            str(row["claim"]),
                            str(row["evidence"])
                        )
                        prediction = result["status"]
                        error_message = ""
                    except Exception as error:
                        prediction = "Verification Error"
                        error_message = str(error)

                    predictions.append(prediction)
                    evaluation_rows.append({
                        "Claim": str(row["claim"]),
                        "Gold Label": str(row["gold_label"]).strip(),
                        "Predicted Label": prediction,
                        "Error": error_message
                    })

            metrics = None
            valid_indices = [
                i for i, prediction in enumerate(predictions)
                if prediction in {"Supported", "Contradicted", "Unsupported"}
            ]
            if not valid_indices:
                st.error("No examples were successfully classified.")
            else:
                y_true = [
                    str(evaluation_data.iloc[i]["gold_label"]).strip()
                    for i in valid_indices
                ]
                y_pred = [predictions[i] for i in valid_indices]

                try:
                    metrics = calculate_metrics(y_true, y_pred)
                except ValueError as error:
                    st.error(f"Evaluation labels are invalid: {error}")
                    metrics = None

                if metrics is not None:
                    st.metric(
                        "Accuracy (successful rows)",
                        f"{metrics['accuracy']:.3f}"
                    )
                    st.metric(
                        "Macro F1",
                        f"{metrics['macro_f1']:.3f}"
                    )
                    st.caption(
                        f"Successfully classified {len(valid_indices)} of "
                        f"{len(evaluation_data)} rows. Errors are excluded from metrics."
                    )

            if metrics is not None:
                st.subheader("Class-wise Performance")
                st.dataframe(
                    pd.DataFrame(metrics["per_class"]).T,
                    use_container_width=True
                )

                st.subheader("Confusion Matrix")
                st.dataframe(
                    pd.DataFrame(
                        metrics["confusion_matrix"],
                        index=metrics["labels"],
                        columns=metrics["labels"]
                    ),
                    use_container_width=True
                )

            st.subheader("Per-example results")
            evaluation_report = pd.DataFrame(evaluation_rows)
            st.dataframe(evaluation_report, use_container_width=True, hide_index=True)
            st.download_button(
                "Download evaluation results",
                evaluation_report.to_csv(index=False),
                "veritrag_evaluation_results.csv",
                "text/csv"
            )


st.divider()

st.caption(
    "VeriRAG is a research prototype. Verification results are evidence-based assessments, not guarantees of factual truth."
)
