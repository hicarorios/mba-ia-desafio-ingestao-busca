import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

BATCH_SIZE = 5
BATCH_DELAY_SECONDS = 1.0

load_dotenv()

REQUIRED_ENV = (
    "GOOGLE_API_KEY",
    "GOOGLE_EMBEDDING_MODEL",
    "DATABASE_URL",
    "PG_VECTOR_COLLECTION_NAME",
    "PDF_PATH",
)


def _check_env() -> None:
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Variáveis de ambiente obrigatórias não definidas: {', '.join(missing)}"
        )


def ingest_pdf() -> None:
    _check_env()

    pdf_path = Path(os.getenv("PDF_PATH"))
    if not pdf_path.is_absolute():
        pdf_path = Path(__file__).resolve().parent.parent / pdf_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    docs = PyPDFLoader(str(pdf_path)).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    if not chunks:
        raise RuntimeError("Nenhum chunk gerado a partir do PDF.")

    enriched = [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)},
        )
        for chunk in chunks
    ]
    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    )

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    total = len(enriched)
    for start in range(0, total, BATCH_SIZE):
        batch_docs = enriched[start : start + BATCH_SIZE]
        batch_ids = ids[start : start + BATCH_SIZE]
        store.add_documents(documents=batch_docs, ids=batch_ids)
        end = start + len(batch_docs)
        print(f"  inseridos {end}/{total} chunks")
        if end < total:
            time.sleep(BATCH_DELAY_SECONDS)

    print(
        f"Ingestão concluída: {total} chunks inseridos em "
        f"'{os.getenv('PG_VECTOR_COLLECTION_NAME')}'."
    )


if __name__ == "__main__":
    ingest_pdf()
