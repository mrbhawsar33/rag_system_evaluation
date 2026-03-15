import ollama
import json

from src.llm.structured_output import RAGAnswer


class OllamaClient:

    def __init__(self, model_name="mistral", prompt_template=None):
        self.model_name = model_name
        self.prompt_template = prompt_template

    def generate(self, query: str, context: str):

        prompt = f"""
{self.prompt_template}

Context:
{context}

Question:
{query}
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        raw_output = response["message"]["content"]

        try:
            data = json.loads(raw_output)
            validated = RAGAnswer(**data)
            return validated

        except Exception:
            return RAGAnswer(
                answer=raw_output.strip(),
                citations=[]
            )