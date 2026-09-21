import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from generation.prompt_builder import PromptMessages

load_dotenv()


@dataclass
class Answer:
    text: str
    citations: List[str]   # extracted citation strings from the answer text
    query: str
    chunks_used: int


def generate(prompt: PromptMessages, query: str, chunks_used: int) -> Answer:
    """
    Call the OpenAI API with the prompt messages.
    Parse citation strings from the response (lines matching '[Source: ..., Section: ...]').
    Return an Answer dataclass.
    Model: gpt-4o-mini
    Max tokens: 1024
    Read OPENAI_API_KEY from environment variables using python-dotenv.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ],
    )

    text = response.choices[0].message.content or ""
    citations = re.findall(r'\[Source: .+?, Section: .+?\]', text)

    return Answer(
        text=text,
        citations=citations,
        query=query,
        chunks_used=chunks_used,
    )
