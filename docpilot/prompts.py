DIRECT = """You are DocPilot, an assistant that answers questions about the user's uploaded documents.
Reply briefly and helpfully to: {question}"""

GRADE = """You check which retrieved passages help answer a question.

Question: {question}

Passages:
{passages}

Reply with JSON only: {{"relevant": [list of passage numbers that contain information useful for the answer]}}"""

REWRITE = """The search below found nothing useful. Rewrite the question as a better search query:
expand abbreviations, add likely keywords and synonyms, and remove filler words.

Original question: {question}
Previous query: {query}

Reply with JSON only: {{"query": "..."}}"""

GENERATE = """Answer the question using ONLY the numbered sources below.
- Cite every claim with the source number in square brackets, like [1] or [2][3].
- If the sources do not contain the answer, say you could not find it in the documents.
- Be concise.

Sources:
{context}

Question: {question}

Answer:"""

GROUNDED = """Check whether every claim in the answer is supported by the sources.

Sources:
{context}

Answer:
{answer}

Reply with JSON only: {{"grounded": true or false, "unsupported": "claims not backed by the sources, if any"}}"""

STRICT_SUFFIX = """

Your previous answer included claims not found in the sources ({unsupported}).
Rewrite it using only what the sources state."""
