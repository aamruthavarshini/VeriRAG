
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
    "Detect unsupported claims, contradictions and citation problems in RAG-generated answers."
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

                answer = ollama.generate_answer(
                    question,
                    passages
                )

                st.session_state.answer = answer

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

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Claim {index}: {result['status']}"
            ):

                st.write(
                    "**Claim:**",
                    result["claim"]
                )

                st.write(
                    "**Explanation:**",
                    result["explanation"]
                )

                st.write(
                    "**Confidence:**",
                    round(result["confidence"], 3)
                )

                st.write(
                    "**Similarity:**",
                    round(result["similarity"], 3)
                )

                st.write(
                    "**Numerical check:**",
                    result["numeric_check"]
                )

                st.write(
                    "**Evidence:**"
                )

                for evidence in result["evidence"]:

                    st.markdown(
                        f"**{evidence['source']} — "
                        f"Passage {evidence['chunk_id']}**"
                    )

                    st.write(
                        evidence["text"]
                    )

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

        evaluation_data = pd.read_csv(
            file
        )

        required = {
            "claim",
            "evidence",
            "gold_label"
        }

        if not required.issubset(
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

            with st.spinner(
                "Evaluating examples..."
            ):

                for _, row in evaluation_data.iterrows():

                    result = verifier.classify_pair(
                        str(row["claim"]),
                        str(row["evidence"])
                    )

                    predictions.append(
                        result["status"]
                    )

            metrics = calculate_metrics(
                evaluation_data["gold_label"].astype(str).tolist(),
                predictions
            )

            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.3f}"
            )

            st.subheader(
                "Class-wise Performance"
            )

            st.dataframe(
                pd.DataFrame(
                    metrics["per_class"]
                ).T,
                use_container_width=True
            )

            st.subheader(
                "Confusion Matrix"
            )

            st.dataframe(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=metrics["labels"],
                    columns=metrics["labels"]
                ),
                use_container_width=True
            )


st.divider()

st.caption(
    "VeriRAG is a research prototype. Verification results are evidence-based assessments, not guarantees of factual truth."
)
