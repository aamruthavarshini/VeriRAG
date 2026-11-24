import re
import fitz


def clean_text(text):
    text = text.replace("\ufeff", "").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, chunk_size=850, overlap=140):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
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


def _decode_text(raw):
    # utf-8-sig removes a BOM; cp1252 is a practical fallback for Windows text.
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw.decode("cp1252")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")


def extract_documents(uploaded_files):
    documents = []
    file_errors = []

    for file in uploaded_files:
        name = getattr(file, "name", "unnamed")
        raw = file.getvalue()
        extension = name.lower().rsplit(".", 1)[-1] if "." in name else ""

        try:
            if extension == "pdf":
                pages = []
                with fitz.open(stream=raw, filetype="pdf") as pdf:
                    for page_number, page in enumerate(pdf, start=1):
                        page_text = page.get_text("text")
                        if page_text.strip():
                            pages.append(f"[Page {page_number}]\n{page_text}")
                text = "\n\n".join(pages)
                if not text.strip():
                    file_errors.append(
                        f"{name}: no extractable text (possibly a scanned/image-only PDF)."
                    )
                    continue
            else:
                text = _decode_text(raw)

            text = clean_text(text)
            chunks = chunk_text(text)
            if not chunks:
                file_errors.append(f"{name}: file is empty or contains no readable text.")
                continue

            for index, chunk in enumerate(chunks, start=1):
                documents.append({
                    "source": name,
                    "chunk_id": index,
                    "text": chunk
                })

        except Exception as error:
            file_errors.append(f"{name}: could not be processed ({error}).")

    # Keep the established list return type for existing app integrations.
    # The UI can read this attribute for user-visible skipped-file messages.
    extract_documents.last_errors = file_errors
    return documents


extract_documents.last_errors = []
