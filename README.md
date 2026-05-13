# Desafio MBA — Ingestão e Busca Semântica (LangChain + Postgres/pgVector)

Software CLI que ingere um PDF em um Postgres com pgVector e responde perguntas usando RAG com Google Gemini.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API do Google (Gemini): https://aistudio.google.com/app/apikey

## Configuração

1. Crie e ative um virtualenv:

   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instale as dependências:

   ```
   pip install -r requirements.txt
   ```

3. Copie `.env.example` para `.env` e preencha `GOOGLE_API_KEY`:

   ```
   cp .env.example .env
   ```

   Variáveis disponíveis:

   | Variável | Descrição | Default sugerido |
   | --- | --- | --- |
   | `GOOGLE_API_KEY` | Chave da API do Google AI Studio | _(obrigatório)_ |
   | `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings | `models/text-embedding-004` |
   | `GOOGLE_LLM_MODEL` | Modelo de chat | `gemini-2.5-flash-lite` |
   | `DATABASE_URL` | URL do Postgres (driver psycopg3) | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
   | `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector | `mba_desafio_rag` |
   | `PDF_PATH` | Caminho do PDF a ingerir | `document.pdf` |

## Execução

1. Suba o Postgres com pgVector:

   ```
   docker compose up -d
   ```

2. Ingestão do PDF (idempotente — pode rodar várias vezes):

   ```
   python src/ingest.py
   ```

3. Chat no terminal:

   ```
   python src/chat.py
   ```

   Exemplo:

   ```
   Faça sua pergunta (CTRL+C para sair):
   PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
   RESPOSTA: O faturamento foi de 10 milhões de reais.

   PERGUNTA: Quantos clientes temos em 2024?
   RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
   ```

## Estrutura

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf
├── src/
│   ├── ingest.py   # Lê o PDF, faz chunking (1000/150) e grava no pgVector
│   ├── search.py   # Retrieval (k=10) + prompt + LLM (search_prompt(question) -> str)
│   └── chat.py     # REPL no terminal
└── README.md
```
