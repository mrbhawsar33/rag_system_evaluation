from pydantic import BaseModel, Field
from typing import List


class RAGAnswer(BaseModel):
    answer: str = Field(
        description="Final answer to the user question using the provided context."
    )

    citations: List[int] = Field(
        description="List of context indices used to support the answer."
    )