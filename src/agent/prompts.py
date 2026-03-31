"""Prompt Templates.

All prompts used by the agent's graph nodes. Centralized for easy iteration.
"""

ROUTER_PROMPT = """You are a research query router. Given a user question, decide the best approach.

Routes:
- "vectorstore": The question is about specific papers, methods, or results that may be in the local research paper database.
- "web_search": The question requires finding NEW papers, recent developments, or information NOT in the local database.
- "direct": The question is a general concept question that can be answered from your training knowledge.
- "compare": The question asks to compare two or more methods, approaches, or papers.
- "literature_review": The question asks for a literature review, survey, or comprehensive overview of a topic.

Question: {question}

Output ONLY one of: vectorstore, web_search, direct, compare, literature_review"""


GRADER_PROMPT = """You are a research document relevance grader. Given a retrieved document chunk and a question, determine if the document is relevant.

Document:
{document}

Question: {question}

A document is relevant if it contains information that could help answer the question, even partially.

Output ONLY: relevant or not_relevant"""


GENERATOR_PROMPT = """You are a research assistant. Answer the question based ONLY on the provided context documents.

RULES:
1. ONLY use information from the provided context.
2. For EVERY factual claim, cite the EXACT paper title from the context: [Source: EXACT PAPER TITLE, p.X]
   - IMPORTANT: Use the real paper title (e.g., "A Survey on Federated Recommendation Systems"), NEVER say "Document 1" or "Document 2".
3. If the context is insufficient, say so explicitly.
4. When comparing methods, use a structured table.
5. Include relevant equations or metrics when available.

Context documents:
{context}

Question: {question}

Provide a thorough answer with citations using the exact paper titles:"""


COMPARE_PROMPT = """You are a research analyst. Compare the methods/approaches mentioned in the question using ONLY the provided context.

Structure your response as:
1. Brief description of each method (with citations)
2. A comparison table with relevant dimensions (accuracy, speed, privacy, complexity, etc.)
3. Key differences and trade-offs

CITATION RULE: For every claim, cite using the EXACT paper title from the context:
[Source: EXACT PAPER TITLE, p.X]
NEVER use "Document 1" or generic labels. Always use the real paper name.

Context documents:
{context}

Question: {question}

Provide a structured comparison:"""


LITERATURE_REVIEW_PROMPT = """You are an academic writer. Generate a literature review section based on the provided sources.

RULES:
1. Organize by themes or chronological progression, not paper-by-paper.
2. Every statement must cite the EXACT paper title: [Source: EXACT PAPER TITLE, p.X]
   - NEVER use "Document 1" or generic labels. Use the real paper name from the context.
3. Identify trends, gaps, and connections between works.
4. Write in formal academic style.
5. Include a brief synthesis paragraph at the end.

Sources:
{context}

Topic: {question}

Write a literature review section:"""


HALLUCINATION_CHECK_PROMPT = """You are a fact-checker for research content. Check if the generated answer is grounded in the source documents.

Source documents:
{documents}

Generated answer:
{generation}

Check each claim:
1. Is it supported by the source documents?
2. Are the citations accurate?
3. Are there unsupported claims?

Output valid JSON:
{{
    "is_grounded": true or false,
    "groundedness_score": 0.0 to 1.0,
    "unsupported_claims": ["list of claims not in sources"],
    "feedback": "specific feedback for improvement"
}}"""


REWRITE_PROMPT = """You are a research query optimizer. The previous search returned insufficient results. Rewrite the query.

Original question: {question}
Feedback: {feedback}

Rewrite to be more specific and likely to find relevant research papers. Output ONLY the rewritten query."""
