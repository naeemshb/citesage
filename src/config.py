import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Project Root ===
PROJECT_ROOT = Path(__file__).parent.parent

# === LLM Provider ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic | ollama
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# === API Keys ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# === Semantic Scholar ===
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

# === Paths ===
PAPERS_DIR = PROJECT_ROOT / os.getenv("PAPERS_DIR", "data/papers")
CHROMA_DB_DIR = PROJECT_ROOT / os.getenv("CHROMA_DB_DIR", "data/chroma_db")

# === Chunking ===
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# === Retrieval ===
TOP_K_RESULTS = 5

# === Agent ===
MAX_RETRIES = 3


def get_llm():
    """Factory: return a LangChain chat model based on the configured provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, streaming=True)
    elif LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=LLM_MODEL, api_key=ANTHROPIC_API_KEY, streaming=True)
    elif LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def get_embeddings():
    """Factory: return an embedding model. Always uses OpenAI for consistency."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
