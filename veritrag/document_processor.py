
import re
import fitz


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=850, overlap=140):
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def extract_documents(uploaded_files):

    documents = []

    for file in uploaded_files:

        name = file.name
        raw = file.getvalue()

        extension = name.lower().split(".")[-1]

        if extension == "pdf":

            pages = []

            with fitz.open(stream=raw, filetype="pdf") as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    text = page.get_text("text")

                    if text.strip():
                        pages.append(
                            f"[Page {page_number}]\n{text}"
                        )

            text = "\n\n".join(pages)

        else:

            text = raw.decode(
                "utf-8",
                errors="replace"
            )

        text = clean_text(text)

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks, start=1):

            documents.append({

                "source": name,

                "chunk_id": index,

                "text": chunk

            })

    return documents
