"""Retrieval-Augmented Generation engine for technical PDF documents."""

import io
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Set, Tuple, TypedDict, Union

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PageObject, PdfReader
from sklearn.feature_extraction.text import CountVectorizer

import config

logger = logging.getLogger(__name__)

PdfSource = Union[bytes, bytearray, str, Path, BinaryIO]
ScoredChunk = Tuple[Document, float]
ChunkKey = Tuple[Any, Any]

FALLBACK_NOTICE = (
    "Local fallback mode: no API key was provided, so the most relevant passage "
    "from your documents is shown below."
)
LLM_ERROR_NOTICE = (
    "The language model request failed ({error}), so the most relevant passage "
    "from your documents is shown below instead."
)
EMPTY_QUESTION_MESSAGE = "Please enter a question about your documents."
NO_INDEX_MESSAGE = "No document has been indexed yet. Upload a PDF to get started."
NO_MATCH_MESSAGE = (
    "No passage in the indexed documents is relevant enough to answer this question. "
    "Try rephrasing it using terms that appear in the document."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are DocuQuery, an assistant that answers questions about technical documents. "
            "Answer using only the context passages provided below. "
            "Cite the page numbers that support each statement in square brackets, for example [page 4], "
            "and never cite passage numbers. "
            "If the passages do not contain the answer, say that the documents do not cover it. "
            "Write clear, plain sentences without LaTeX and never use em dashes.\n\n"
            "Context passages:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You help a search engine find passages in technical documents such as standards, manuals, and syllabi. "
            "Write one short paragraph, under 80 words, that reads like the passage in such a document that answers "
            "the question. Use the precise terminology, identifiers, function signatures, and table or section names "
            "that document would most likely contain. If such documents usually state the answer in a general "
            "requirements table or a shared section that covers many items, rather than in the entry for the specific "
            "item, write the paragraph the way that general table or section would. Output only the paragraph.",
        ),
        ("human", "{question}"),
    ]
)

_MARKDOWN_SPECIAL_CHARACTERS = re.compile(r"([\\`*_\[\]<>$~|])")
_DOT_LEADER_LINE = re.compile(r"(?:\.\s*){4,}\d+\s*$")
_CONTENTS_MIN_LINES = 5
_CONTENTS_LEADER_SHARE = 0.5
_KEYWORD_TOKEN_PATTERN = r"(?u)\w+"
_CITATION_BRACKETS = str.maketrans({"【": "[", "】": "]"})
_LATEX_BLOCK = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_LATEX_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)


class DocumentProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


class SourceCitation(TypedDict):
    page_number: int
    chunk_id: int
    snippet: str
    source: str
    score: float


class QueryResult(TypedDict):
    answer: str
    sources: List[SourceCitation]


def create_embeddings(model_name: str = config.EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Create the sentence-transformers embedding model with unit-normalized output vectors."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


class KeywordIndex:
    """Okapi BM25 index that rewards exact terms such as unordered_map, a_uniq, or Table 91."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._weights: Optional[Any] = None
        self._analyzer: Optional[Any] = None
        self._vocabulary: Dict[str, int] = {}
        vectorizer = CountVectorizer(token_pattern=_KEYWORD_TOKEN_PATTERN, lowercase=True)
        try:
            counts = vectorizer.fit_transform(texts).tocsr().astype(np.float32)
        except ValueError:
            logger.warning("Keyword index is empty because the chunks contain no searchable words")
            return

        document_count = counts.shape[0]
        document_lengths = np.asarray(counts.sum(axis=1)).ravel()
        average_length = max(float(document_lengths.mean()), 1.0)
        document_frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        inverse_frequency = np.log1p((document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        rows = np.repeat(np.arange(document_count), np.diff(counts.indptr))
        term_frequency = counts.data
        length_penalty = k1 * (1.0 - b + b * document_lengths[rows] / average_length)
        counts.data = (
            inverse_frequency[counts.indices] * term_frequency * (k1 + 1.0) / (term_frequency + length_penalty)
        ).astype(np.float32)

        self._weights = counts.tocsc()
        self._analyzer = vectorizer.build_analyzer()
        self._vocabulary = vectorizer.vocabulary_

    def rank(self, query: str, limit: int) -> List[int]:
        """Return the positions of up to limit texts that share at least one term with the query, best first."""
        if self._weights is None or self._analyzer is None:
            return []
        columns = sorted({self._vocabulary[token] for token in self._analyzer(query) if token in self._vocabulary})
        if not columns:
            return []
        scores = np.asarray(self._weights[:, columns].sum(axis=1)).ravel()
        candidates = np.flatnonzero(scores > 0)
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [int(position) for position in ordered[:limit]]


class RAGEngine:
    """Loads PDFs, indexes them for hybrid search, and answers questions with page-level citations."""

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        embedding_model_name: str = config.EMBEDDING_MODEL_NAME,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        top_k: int = config.TOP_K,
        min_relevance: float = config.MIN_RELEVANCE_SCORE,
        llm_model: str = config.DEFAULT_LLM_MODEL,
        candidate_count: int = config.RETRIEVAL_CANDIDATES,
        query_expansion: bool = config.QUERY_EXPANSION,
        context_neighbors: int = config.CONTEXT_NEIGHBORS,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.llm_model = llm_model
        self.candidate_count = candidate_count
        self.query_expansion = query_expansion
        self.context_neighbors = context_neighbors
        self.vector_store: Optional[FAISS] = None
        self.chunk_count: int = 0
        self._embeddings = embeddings
        self._chunks: List[Document] = []
        self._positions: Dict[ChunkKey, int] = {}
        self._chunk_vectors: Optional[np.ndarray] = None
        self._keyword_index: Optional[KeywordIndex] = None
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            logger.info("Loading embedding model %s", self.embedding_model_name)
            self._embeddings = create_embeddings(self.embedding_model_name)
        return self._embeddings

    @property
    def is_ready(self) -> bool:
        return self.vector_store is not None

    def load_pdf(self, file_bytes_or_path: PdfSource, source_name: Optional[str] = None) -> List[Document]:
        """Extract one Document per page with text, tagged with its source name and 1-based page number."""
        name = source_name or _describe_source(file_bytes_or_path)
        try:
            reader = PdfReader(_as_readable(file_bytes_or_path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise DocumentProcessingError(f"{name} is password protected and cannot be read.")
            pages = _extract_pages(reader, name)
            total_pages = len(reader.pages)
        except DocumentProcessingError as error:
            logger.error("PDF processing failed: %s", error)
            raise
        except FileNotFoundError as error:
            logger.error("PDF file not found: %s", name)
            raise DocumentProcessingError(f"File not found: {name}") from error
        except Exception as error:
            logger.exception("Unable to parse %s as a PDF", name)
            raise DocumentProcessingError(f"{name} could not be read as a PDF ({error}).") from error

        if not pages:
            message = (
                f"{name} contains no extractable text. "
                "Scanned or image based PDFs need OCR before they can be indexed."
            )
            logger.error("PDF processing failed: %s", message)
            raise DocumentProcessingError(message)

        logger.info("Extracted text from %d of %d pages in %s", len(pages), total_pages, name)
        return [
            Document(page_content=text, metadata={"source": name, "page_number": page_number})
            for page_number, text in pages
        ]

    def chunk_text(self, documents: List[Document]) -> List[Document]:
        """Split page documents into overlapping chunks with sequential chunk ids."""
        chunks = self._splitter.split_documents(documents)
        for chunk_id, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = chunk_id
        logger.info("Split %d pages into %d chunks", len(documents), len(chunks))
        return chunks

    def build_vector_store(self, chunks: List[Document]) -> FAISS:
        """Embed the chunks into a fresh FAISS store and build the matching keyword index."""
        if not chunks:
            raise ValueError("Cannot build a vector store from an empty list of chunks.")
        texts = [chunk.page_content for chunk in chunks]
        vectors = np.asarray(self.embeddings.embed_documents(texts), dtype=np.float32)
        self.vector_store = FAISS.from_embeddings(
            list(zip(texts, vectors.tolist())),
            self.embeddings,
            metadatas=[dict(chunk.metadata) for chunk in chunks],
            ids=[str(position) for position in range(len(chunks))],
        )
        self._chunks = list(chunks)
        keys = [_chunk_key(chunk) for chunk in chunks]
        self._positions = dict(zip(keys, range(len(keys)))) if len(set(keys)) == len(keys) else {}
        self._chunk_vectors = vectors
        self._keyword_index = KeywordIndex(texts)
        self.chunk_count = len(chunks)
        logger.info("Indexed %d chunks with %s and BM25", self.chunk_count, self.embedding_model_name)
        return self.vector_store

    def retrieve(self, question: str, expansion: str = "") -> List[ScoredChunk]:
        """Fuse FAISS and BM25 rankings and return up to top_k chunks that clear the relevance threshold.

        The optional expansion text is added to the search query, while the relevance score of each chunk
        is always its cosine similarity to the question itself.
        """
        if self.vector_store is None or self._chunk_vectors is None or self._keyword_index is None:
            return []
        search_text = f"{question}\n{expansion}" if expansion else question
        question_vector = np.asarray(self.embeddings.embed_query(question), dtype=np.float32)
        search_vector = (
            np.asarray(self.embeddings.embed_query(search_text), dtype=np.float32) if expansion else question_vector
        )

        dense_hits = self.vector_store.similarity_search_with_score_by_vector(
            search_vector.tolist(), k=self.candidate_count
        )
        dense_ranking = [int(document.id) for document, _ in dense_hits if document.id is not None]
        keyword_ranking = self._keyword_index.rank(search_text, self.candidate_count)

        matches: List[ScoredChunk] = []
        for position in _reciprocal_rank_fusion([dense_ranking, keyword_ranking], config.RRF_K):
            score = float(self._chunk_vectors[position] @ question_vector)
            if score >= self.min_relevance:
                matches.append((self._chunks[position], score))
            if len(matches) == self.top_k:
                break
        return matches

    def query(self, user_question: str, api_key: Optional[str] = None) -> QueryResult:
        """Answer a question from the indexed documents, using the LLM when an API key is supplied."""
        question = (user_question or "").strip()
        if not question:
            return QueryResult(answer=EMPTY_QUESTION_MESSAGE, sources=[])
        if self.vector_store is None:
            return QueryResult(answer=NO_INDEX_MESSAGE, sources=[])

        key = (api_key or "").strip()
        expansion = self._expand_query(question, key) if key and self.query_expansion else ""
        matches = self.retrieve(question, expansion)
        if not matches:
            logger.info("No chunk reached the relevance threshold for: %s", question)
            return QueryResult(answer=NO_MATCH_MESSAGE, sources=[])

        sources = [_to_citation(document, score) for document, score in matches]
        best_match = matches[0][0]
        if not key:
            return QueryResult(answer=_fallback_answer(best_match, FALLBACK_NOTICE), sources=sources)

        try:
            answer = self._generate_llm_answer(question, matches, key)
        except Exception as error:
            logger.exception("LLM generation failed with model %s", self.llm_model)
            answer = _fallback_answer(best_match, LLM_ERROR_NOTICE.format(error=type(error).__name__))
        return QueryResult(answer=answer, sources=sources)

    def _create_llm(self, api_key: str, timeout: float, max_retries: int) -> ChatGroq:
        return ChatGroq(
            model=self.llm_model,
            groq_api_key=api_key,
            temperature=config.LLM_TEMPERATURE,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _expand_query(self, question: str, api_key: str) -> str:
        try:
            llm = self._create_llm(api_key, config.QUERY_EXPANSION_TIMEOUT_SECONDS, max_retries=0)
            chain = QUERY_EXPANSION_PROMPT | llm | StrOutputParser()
            expansion = chain.invoke({"question": question}).strip()
        except Exception:
            logger.warning("Query expansion failed, searching with the question alone", exc_info=True)
            return ""
        logger.debug("Expanded search query for %r: %s", question, expansion)
        return expansion

    def _generate_llm_answer(self, question: str, matches: List[ScoredChunk], api_key: str) -> str:
        llm = self._create_llm(api_key, config.LLM_TIMEOUT_SECONDS, config.LLM_MAX_RETRIES)
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": self._build_context(matches), "question": question}).strip()
        if not answer:
            raise ValueError("The language model returned an empty answer.")
        return _tidy_answer(answer)

    def _build_context(self, matches: List[ScoredChunk]) -> str:
        """Number each retrieved chunk together with its neighbors from the same page, so split table rows stay whole."""
        blocks: List[str] = []
        included: Set[int] = set()
        for document, _ in matches:
            position = self._positions.get(_chunk_key(document))
            if position is None:
                window = [document]
            elif position in included:
                continue
            else:
                nearby = [
                    candidate
                    for candidate in range(position - self.context_neighbors, position + self.context_neighbors + 1)
                    if 0 <= candidate < len(self._chunks)
                    and candidate not in included
                    and _same_page(self._chunks[candidate], document)
                ]
                included.update(nearby)
                window = [self._chunks[candidate] for candidate in nearby]
            text = "\n".join(chunk.page_content for chunk in window)
            blocks.append(
                f"Passage {len(blocks) + 1} ({document.metadata.get('source', 'document')}, "
                f"page {document.metadata.get('page_number', 'unknown')}):\n{text}"
            )
        return "\n\n".join(blocks)


def _as_readable(source: PdfSource) -> Union[Path, BinaryIO]:
    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise DocumentProcessingError("The uploaded file is empty.")
        return io.BytesIO(source)
    if isinstance(source, (str, Path)):
        return Path(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return source


def _describe_source(source: PdfSource) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    return getattr(source, "name", None) or "uploaded document"


def _extract_pages(reader: PdfReader, name: str) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    contents_pages: List[Tuple[int, str]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        cleaned = _normalize_whitespace(_extract_page_text(page, page_number, name))
        if not cleaned:
            continue
        if _is_contents_page(cleaned):
            contents_pages.append((page_number, cleaned))
        else:
            pages.append((page_number, cleaned))
    if not pages:
        return contents_pages
    if contents_pages:
        logger.info("Skipped %d table of contents pages in %s", len(contents_pages), name)
    return pages


def _extract_page_text(page: PageObject, page_number: int, name: str) -> str:
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except Exception:
        logger.debug("Layout extraction failed on page %d of %s, using plain mode", page_number, name, exc_info=True)
    try:
        return page.extract_text() or ""
    except Exception:
        logger.warning("Skipping page %d of %s: text extraction failed", page_number, name, exc_info=True)
        return ""


def _normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _is_contents_page(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < _CONTENTS_MIN_LINES:
        return False
    leader_lines = sum(1 for line in lines if _DOT_LEADER_LINE.search(line))
    return leader_lines / len(lines) >= _CONTENTS_LEADER_SHARE


def _reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int) -> List[int]:
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            fused[position] = fused.get(position, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.__getitem__, reverse=True)


def _to_citation(document: Document, score: float) -> SourceCitation:
    return SourceCitation(
        page_number=int(document.metadata.get("page_number", 0)),
        chunk_id=int(document.metadata.get("chunk_id", -1)),
        snippet=document.page_content,
        source=str(document.metadata.get("source", "")),
        score=round(score, 4),
    )


def _tidy_answer(answer: str) -> str:
    """Turn full width citation brackets into [ ] and LaTeX delimiters into the $ form that Streamlit renders."""
    answer = answer.translate(_CITATION_BRACKETS)
    answer = _LATEX_BLOCK.sub(lambda match: f"$${match.group(1).strip()}$$", answer)
    return _LATEX_INLINE.sub(lambda match: f"${match.group(1).strip()}$", answer)


def _chunk_key(document: Document) -> ChunkKey:
    return document.metadata.get("source"), document.metadata.get("chunk_id")


def _same_page(first: Document, second: Document) -> bool:
    return (
        first.metadata.get("source") == second.metadata.get("source")
        and first.metadata.get("page_number") == second.metadata.get("page_number")
    )


def _fallback_answer(document: Document, notice: str) -> str:
    page_number = document.metadata.get("page_number", "unknown")
    return f"{notice}\n\n**Best matching passage, page {page_number}:**\n\n{_as_markdown_quote(document.page_content)}"


def _as_markdown_quote(text: str) -> str:
    escaped = _MARKDOWN_SPECIAL_CHARACTERS.sub(r"\\\1", text)
    return "\n".join(f"> {line}" if line else ">" for line in escaped.splitlines())
