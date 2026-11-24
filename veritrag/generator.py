
import requests


class OllamaClient:

    def __init__(
        self,
        base_url="http://localhost:11434",
        model="llama3.2:latest"
    ):

        self.base_url = base_url.rstrip("/")
        self.model = model

    def check(self):

        try:

            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=8
            )

            response.raise_for_status()

            models = [
                item["name"]
                for item in response.json().get("models", [])
            ]

            return True, models

        except Exception as error:

            return False, str(error)

    def chat(self, prompt, system=None):

        messages = []

        if system:

            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {

            "model": self.model,

            "messages": messages,

            "stream": False,

            "options": {
                "temperature": 0.1
            }

        }

        try:

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180
            )

            response.raise_for_status()

            return response.json()["message"]["content"]

        except requests.exceptions.ConnectionError:

            raise RuntimeError(
                "Cannot connect to Ollama. Start Ollama first."
            )

        except requests.exceptions.Timeout:

            raise RuntimeError(
                "Ollama timed out. Try a smaller model."
            )

        except Exception as error:

            raise RuntimeError(str(error))

    def generate_answer(self, question, passages):

        context = ""

        for passage in passages:

            context += (
                f"\n[Source: {passage['source']} | "
                f"Passage: {passage['chunk_id']}]\n"
                f"{passage['text']}\n"
            )

        system = """
        You are a careful document question-answering assistant.

        Answer using only the supplied source documents.

        Never invent facts, dates or numerical values.

        If information is unavailable, say so.

        Cite factual statements using:

        [Source: filename | Passage: number]
        """

        prompt = f"""
        QUESTION:

        {question}

        SOURCE DOCUMENTS:

        {context}

        Generate a concise answer with citations.
        """

        return self.chat(prompt, system)
