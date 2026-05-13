import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

REQUIRED_ENV = (
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL",
    "GOOGLE_LLM_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
)


def _check_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Variáveis de ambiente obrigatórias não definidas: {', '.join(missing)}"
        )


_check_env()

_embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
_store = PGVector(
    embeddings=_embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)
_llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"), temperature=0)
_chain = PromptTemplate.from_template(PROMPT_TEMPLATE) | _llm


def search_prompt(question: str) -> str:
    if not question or not question.strip():
        return FALLBACK_ANSWER

    results = _store.similarity_search_with_score(question, k=10)
    if not results:
        return FALLBACK_ANSWER

    contexto = "\n\n".join(doc.page_content for doc, _ in results)
    response = _chain.invoke({"contexto": contexto, "pergunta": question})
    answer = (response.content if hasattr(response, "content") else str(response)).strip()
    return answer or FALLBACK_ANSWER
