# DocuQuery: Transformer-Based RAG Technical Document Assistant

DocuQuery is a Retrieval-Augmented Generation (RAG) system for technical PDFs such as C++ standard drafts, API manuals, and university syllabi. Upload documents through a Streamlit web interface, ask questions in natural language, and get answers backed by page level source citations.

## Features

- Layout aware PDF ingestion with pypdf: table rows stay on one line, ligatures such as "ﬁ" become plain letters, and table of contents pages are skipped. Empty, encrypted, scanned, and corrupted files are handled gracefully.
- Recursive character chunking: 500 character chunks with 50 characters of overlap.
- 384 dimensional dense embeddings from `sentence-transformers/all-MiniLM-L6-v2`, indexed in FAISS.
- Hybrid retrieval: FAISS semantic search and BM25 keyword search are fused with reciprocal rank fusion, so exact identifiers such as `a_uniq.insert(t)` or "Table 91" are found as reliably as paraphrased questions.
- Query expansion: with an API key, the model first drafts a short passage in the document's own terminology, which is added to the search query.
- A relevance threshold, so unrelated questions get a clear "no match" reply instead of noise.
- Optional answer synthesis with an open model served by Groq through LangChain (`langchain-groq`).
- Deterministic local fallback: without an API key, the best matching passage is returned verbatim with a clear notice.
- Source citations with file name, page number, chunk id, and relevance score in collapsible panels.
- Multiple PDFs per session, with the vector index and the conversation kept in Streamlit session state.

## How It Works

1. **Load.** pypdf extracts every page in layout mode, which keeps each table row on one line. The text is normalized with Unicode NFKC, and pages that are mostly dot leader entries (a table of contents) are skipped. Each remaining page becomes a LangChain `Document` tagged with its file name and page number.
2. **Chunk.** `RecursiveCharacterTextSplitter` splits pages into overlapping chunks, and each chunk receives a sequential `chunk_id`.
3. **Index.** Chunks are encoded into unit length 384 dimensional vectors with `HuggingFaceEmbeddings` and stored in a FAISS index. A BM25 keyword index is built over the same chunks.
4. **Expand.** With a Groq API key, `ChatGroq` writes a short passage that reads like the answer would in the document. It only steers the search: it is never shown to the user or sent as context.
5. **Retrieve.** FAISS (`similarity_search_with_score_by_vector`) and BM25 each return their best `RETRIEVAL_CANDIDATES` chunks for the question plus the expansion. The two rankings are merged with reciprocal rank fusion, and the first `TOP_K` chunks whose cosine similarity to the question reaches `MIN_RELEVANCE_SCORE` are kept.
6. **Answer.** With a Groq API key, each retrieved chunk is sent to `ChatGroq` together with its neighboring chunks from the same page, so a table row that was split between two chunks reaches the model whole. The passages are labeled with their page numbers, and the model is told to answer only from them and cite pages. Without a key, or when the API call fails, the top chunk is returned directly.

Each question with a key costs two Groq requests: one for the expansion and one for the answer. If the expansion fails or times out, DocuQuery searches with the question alone and still answers.

`RAGEngine.query()` returns a dictionary with this shape:

```python
{
    "answer": "Local fallback mode: no API key was provided, ...",
    "sources": [
        {
            "page_number": 12,
            "chunk_id": 48,
            "snippet": "std::unordered_map is an associative container ...",
            "source": "cpp_reference.pdf",
            "score": 0.71,
        }
    ],
}
```

## Project Structure

- `config.py`: centralized settings for chunking, embeddings, retrieval, and the language model.
- `rag_engine.py`: the `RAGEngine` class covering PDF loading, chunking, FAISS indexing, retrieval, and answer generation.
- `app.py`: Streamlit dashboard with document upload, status panel, chat interface, and citations.
- `requirements.txt`: pinned dependencies.
- `tests/test_rag_engine.py`: pytest unit and integration tests.
- `.streamlit/config.toml`: neutral light theme with white backgrounds, black text, and light gray borders.
- `.env.example`: template for optional environment variables.

## Requirements

- Python 3.11 or newer, required by the pinned Streamlit and PyTorch releases. The full test suite was verified on Python 3.14 on Windows 11.
- Around 1.5 GB of disk space for dependencies, mostly the CPU build of PyTorch.
- Internet access on the first run, to download the embedding model (about 90 MB) from Hugging Face. It is cached locally afterwards.

## Setup

1. Open a terminal in the project folder.

   ```
   cd NLP-PROJECT-RAG
   ```

2. Create a virtual environment.

   ```
   python -m venv venv
   ```

3. Activate it.

   Windows PowerShell:

   ```
   .\venv\Scripts\Activate.ps1
   ```

   Windows Command Prompt:

   ```
   venv\Scripts\activate.bat
   ```

   macOS or Linux:

   ```
   source venv/bin/activate
   ```

   If PowerShell refuses to run the activation script, allow local scripts for your user once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and try again.

4. Install the dependencies.

   ```
   pip install -r requirements.txt
   ```

5. Optional: store your Groq API key so you do not need to paste it into the sidebar each time. Create a free key at https://console.groq.com/keys, then copy the template:

   ```
   copy .env.example .env
   ```

   Use `cp .env.example .env` on macOS or Linux, then set `GROQ_API_KEY` in `.env`. Only `.env` is read by the app and only `.env` is listed in `.gitignore`, so keep real keys out of `.env.example`, which is meant to be committed.

## Running the App

```
streamlit run app.py
```

Streamlit opens the dashboard at http://localhost:8501.

1. Upload one or more PDFs in the sidebar. Indexing starts automatically and the status panel shows the document count, chunk count, and index status.
2. Optionally paste a Groq API key. Without one, DocuQuery runs in local fallback mode and needs no network access after the model download.
3. Ask questions in the chat box. Open **📌 Source Passages** under any answer to read the cited passages with their page numbers.

## Running the Tests

```
pytest
```

For one line per test:

```
pytest -v
```

The suite covers:

- `test_pdf_processing`: text extraction from a generated two page PDF, page metadata, chunk length bounds, and sequential chunk ids.
- `test_embedding_dimensions`: query and document embeddings, plus the FAISS index, have exactly 384 dimensions.
- `test_faiss_retrieval`: a small C++ and syllabus knowledge base where each question must retrieve its exact target chunk as the top result, including the `std::unordered_map` complexity rules.
- `test_fallback_mechanism`: queries without an API key, or with a blank one, return the verbatim top passage with the fallback notice and never call the LLM.
- Retrieval quality tests: BM25 ranks exact identifiers first, table of contents pages are skipped, ligatures are normalized, and the model's context includes same page neighbors of each retrieved chunk but nothing from other pages.
- Error handling tests for empty and corrupted PDFs (including the error log), missing files, empty questions, a missing index, questions with no relevant match, LLM failures, and failed query expansion.

The Groq tests replace `ChatGroq` with a recording stand in that checks the model name, the API key, that the expansion reaches the search, and that every retrieved passage reaches the answer prompt, so no API key or network call is needed. The first test run downloads the embedding model if it is not cached yet.

## Configuration

All settings live in `config.py`:

- `CHUNK_SIZE = 500`: maximum characters per chunk.
- `CHUNK_OVERLAP = 50`: characters shared between consecutive chunks.
- `EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"`: sentence-transformers model used for embeddings.
- `EMBEDDING_DIMENSION = 384`: output dimension of the embedding model.
- `TOP_K = 3`: number of chunks retrieved per question.
- `MIN_RELEVANCE_SCORE = 0.2`: minimum cosine similarity to the question a chunk needs to be used as a source.
- `RETRIEVAL_CANDIDATES = 50`: chunks each search method contributes before the rankings are fused.
- `RRF_K = 60`: smoothing constant of reciprocal rank fusion.
- `QUERY_EXPANSION = True`: whether the LLM drafts a search passage before retrieval when an API key is available.
- `QUERY_EXPANSION_TIMEOUT_SECONDS = 10.0`: time limit for the expansion request, which is never retried.
- `CONTEXT_NEIGHBORS = 1`: chunks before and after each retrieved chunk, on the same page, that are added to the model's context. The source citations still show only the retrieved chunks.
- `DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"`: Groq model used when an API key is available. Override it with the `DOCUQUERY_LLM_MODEL` environment variable.
- `LLM_TEMPERATURE`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`: generation and network settings for the chat model.

Environment variables, read from the shell or from `.env`:

- `GROQ_API_KEY`: Groq API key used when the sidebar key field is left empty.
- `DOCUQUERY_LLM_MODEL`: overrides the default Groq model. It must be a model ID that Groq currently serves.

## Troubleshooting

- **An answer says the documents do not cover the question, although they do**: the right passage was not among the `TOP_K` retrieved chunks. Open **📌 Source Passages** to see what was retrieved, and name the exact terms the document uses, such as a function name, table number, or section title. On very large documents you can also raise `TOP_K` to 5, which gives the model more passages to read.
- **"contains no extractable text"**: the PDF is scanned or image based. Run it through an OCR tool to add a text layer, then upload it again.
- **"The language model request failed (AuthenticationError)"**: the Groq API key is invalid or revoked. The answer still shows the best matching passage.
- **"The language model request failed (BadRequestError)" or "(NotFoundError)"**: Groq does not serve the configured model. Groq retires models regularly: `llama3-8b-8192`, `llama-3.1-8b-instant`, and `llama-3.3-70b-versatile` are no longer served. Set `DOCUQUERY_LLM_MODEL` in `.env` to a current chat model from https://console.groq.com/docs/models. The default, `openai/gpt-oss-20b`, is the replacement Groq names for its retired 8B Llama models.
- **"The language model request failed (RateLimitError)"**: the free tier request or token limit was reached. Wait a minute and ask again.
- **Slow first question or first upload**: the embedding model loads once per Streamlit server process and is cached afterwards. Very large PDFs, such as a full C++ standard draft, can take a few minutes to embed on a CPU.
- **Offline use**: after the model has been downloaded once, set the environment variable `HF_HUB_OFFLINE=1` to stop Hugging Face from checking for updates.
