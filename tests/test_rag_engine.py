"""Unit and integration tests for the DocuQuery RAG engine."""

import io
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import RunnableLambda

import config
import rag_engine
from rag_engine import (
    EMPTY_QUESTION_MESSAGE,
    FALLBACK_NOTICE,
    NO_INDEX_MESSAGE,
    NO_MATCH_MESSAGE,
    DocumentProcessingError,
    KeywordIndex,
    RAGEngine,
    create_embeddings,
)

UNORDERED_MAP_PARAGRAPH = (
    "The class template std::unordered_map is an associative container that stores key value pairs "
    "in a hash table. Elements are organized into buckets according to the hash of their keys, which "
    "lets search, insertion, and removal run in average constant time. In the worst case, when many "
    "keys collide in the same bucket, these operations degrade to linear time in the size of the "
    "container. Calling rehash or reserve changes the bucket count and invalidates all iterators, "
    "although references to the stored elements remain valid. The max_load_factor member controls "
    "how full the buckets may become before the container grows automatically."
)

SORTING_PARAGRAPH = (
    "The algorithm std::sort arranges the elements of a range in non descending order. Since C++11 "
    "the standard requires O(N log N) comparisons in the worst case, which implementations satisfy "
    "with introsort, a hybrid of quicksort, heapsort, and insertion sort. The sort is not stable, so "
    "elements that compare equal may change their relative order. When stability is required, use "
    "std::stable_sort, which performs O(N log N) comparisons when enough additional memory is "
    "available and O(N log squared N) comparisons otherwise."
)

KNOWLEDGE_BASE: List[Tuple[str, int]] = [
    (
        "std::unordered_map is an associative container implemented as a hash table. Lookup, "
        "insertion, and removal of elements have average constant time complexity O(1), degrading "
        "to linear time O(n) in the worst case when many keys collide in the same bucket.",
        12,
    ),
    (
        "std::map is a sorted associative container that is usually implemented as a red black "
        "tree. Search, removal, and insertion operations have logarithmic complexity O(log n), and "
        "iterating over the map visits the keys in ascending order.",
        14,
    ),
    (
        "std::vector stores its elements contiguously in a dynamically allocated array. Appending "
        "with push_back runs in amortized constant time. When the capacity is exhausted the vector "
        "reallocates its storage, which invalidates all iterators and references to its elements.",
        21,
    ),
    (
        "std::unique_ptr is a smart pointer that exclusively owns a dynamically allocated object and "
        "deletes it when the pointer goes out of scope. A unique_ptr cannot be copied, only moved, "
        "which models single ownership under the RAII idiom.",
        33,
    ),
    (
        "std::sort orders the elements of a range using the introsort algorithm and guarantees "
        "O(N log N) comparisons in the worst case. The algorithm is not stable, so elements that "
        "compare equal may be reordered; use std::stable_sort to preserve their relative order.",
        47,
    ),
    (
        "Course CS 101 grading policy: the final exam is worth 40 percent of the course grade, the "
        "midterm exam 25 percent, weekly programming assignments 30 percent, and class participation "
        "5 percent. Late assignments lose 10 percent per day.",
        2,
    ),
]

RETRIEVAL_CASES: List[Tuple[str, int]] = [
    ("What is the average time complexity of lookups in an unordered_map hash table?", 0),
    ("Which tree data structure does std::map use and how fast is searching it?", 1),
    ("Does push_back on a vector invalidate iterators when it reallocates?", 2),
    ("Can a unique_ptr be copied?", 3),
    ("Is std::sort a stable sorting algorithm?", 4),
    ("How much is the final exam worth in the course grade?", 5),
]

UNORDERED_MAP_QUESTION = RETRIEVAL_CASES[0][0]

CONTENTS_LINES = [
    f"{number}.{section} {title} . . . . . . . . . . . . . . . . {page}"
    for number, section, title, page in [
        (23, 1, "General container requirements", 831),
        (23, 2, "Sequence containers", 840),
        (23, 3, "Associative containers", 851),
        (23, 4, "Unordered associative containers", 862),
        (23, 5, "Container adaptors", 880),
        (24, 1, "Iterator requirements", 901),
    ]
]


def is_expansion_prompt(messages: List[Any]) -> bool:
    return "search engine" in messages[0].content


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: List[List[str]]) -> bytes:
    """Build a minimal, valid PDF in which every inner list becomes one page of text lines."""
    page_object_ids = [4 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{object_id} 0 R" for object_id in page_object_ids)
    bodies: Dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    for object_id, lines in zip(page_object_ids, pages):
        operators = " T* ".join(f"({_escape_pdf_text(line)}) Tj" for line in lines)
        stream = f"BT /F1 10 Tf 14 TL 72 740 Td {operators} ET"
        bodies[object_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {object_id + 1} 0 R >>"
        )
        bodies[object_id + 1] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"

    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets: Dict[int, int] = {}
    for object_id in sorted(bodies):
        offsets[object_id] = buffer.tell()
        buffer.write(f"{object_id} 0 obj\n{bodies[object_id]}\nendobj\n".encode("latin-1"))

    xref_offset = buffer.tell()
    size = len(bodies) + 1
    buffer.write(f"xref\n0 {size}\n0000000000 65535 f \n".encode("latin-1"))
    for object_id in range(1, size):
        buffer.write(f"{offsets[object_id]:010d} 00000 n \n".encode("latin-1"))
    buffer.write(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin-1"))
    return buffer.getvalue()


@pytest.fixture(scope="module")
def sample_pdf_bytes() -> bytes:
    return build_pdf([textwrap.wrap(UNORDERED_MAP_PARAGRAPH, 80), textwrap.wrap(SORTING_PARAGRAPH, 80)])


@pytest.fixture(scope="module")
def embeddings() -> Embeddings:
    return create_embeddings()


@pytest.fixture(scope="module")
def indexed_engine(embeddings: Embeddings) -> RAGEngine:
    engine = RAGEngine(embeddings=embeddings)
    documents = [
        Document(page_content=text, metadata={"source": "cpp_reference.pdf", "page_number": page_number})
        for text, page_number in KNOWLEDGE_BASE
    ]
    engine.build_vector_store(engine.chunk_text(documents))
    return engine


def test_pdf_processing(tmp_path: Path, sample_pdf_bytes: bytes) -> None:
    pdf_path = tmp_path / "cpp_containers.pdf"
    pdf_path.write_bytes(sample_pdf_bytes)
    engine = RAGEngine()

    documents = engine.load_pdf(pdf_path)

    assert [document.metadata["page_number"] for document in documents] == [1, 2]
    assert all(document.metadata["source"] == "cpp_containers.pdf" for document in documents)
    assert "std::unordered_map" in documents[0].page_content
    assert "average constant time" in documents[0].page_content
    assert "introsort" in documents[1].page_content

    chunks = engine.chunk_text(documents)

    assert len(chunks) > len(documents)
    assert all(0 < len(chunk.page_content) <= config.CHUNK_SIZE for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.metadata["page_number"] for chunk in chunks} == {1, 2}


def test_table_of_contents_pages_are_skipped() -> None:
    engine = RAGEngine()
    content_lines = textwrap.wrap(UNORDERED_MAP_PARAGRAPH, 80)

    documents = engine.load_pdf(build_pdf([CONTENTS_LINES, content_lines]), source_name="standard.pdf")
    contents_only = engine.load_pdf(build_pdf([CONTENTS_LINES]), source_name="contents.pdf")

    assert [document.metadata["page_number"] for document in documents] == [2]
    assert "std::unordered_map" in documents[0].page_content
    assert [document.metadata["page_number"] for document in contents_only] == [1]


def test_text_normalization_expands_ligatures() -> None:
    assert rag_engine._normalize_whitespace("eﬃcient ﬁnd()  and ﬂags") == "efficient find() and flags"


def test_load_pdf_accepts_bytes_and_streams(sample_pdf_bytes: bytes) -> None:
    engine = RAGEngine()

    from_bytes = engine.load_pdf(sample_pdf_bytes, source_name="manual.pdf")
    from_stream = engine.load_pdf(io.BytesIO(sample_pdf_bytes), source_name="manual.pdf")

    assert [document.page_content for document in from_bytes] == [document.page_content for document in from_stream]
    assert all(document.metadata["source"] == "manual.pdf" for document in from_bytes)


@pytest.mark.parametrize(
    "payload",
    [b"", b"this is plain text, not a PDF", build_pdf([[]])],
    ids=["empty_bytes", "not_a_pdf", "no_extractable_text"],
)
def test_load_pdf_rejects_unreadable_input(payload: bytes, caplog: pytest.LogCaptureFixture) -> None:
    engine = RAGEngine()

    with caplog.at_level(logging.ERROR, logger=rag_engine.__name__):
        with pytest.raises(DocumentProcessingError):
            engine.load_pdf(payload, source_name="broken.pdf")

    assert any(record.levelno >= logging.ERROR for record in caplog.records)


def test_load_pdf_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DocumentProcessingError, match="File not found"):
        RAGEngine().load_pdf(tmp_path / "missing.pdf")


def test_build_vector_store_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError):
        RAGEngine().build_vector_store([])


def test_embedding_dimensions(embeddings: Embeddings, indexed_engine: RAGEngine) -> None:
    query_vector = embeddings.embed_query("Average lookup complexity of std::unordered_map")
    document_vectors = embeddings.embed_documents(["std::vector growth policy", "CS 101 grading policy"])

    assert config.EMBEDDING_DIMENSION == 384
    assert len(query_vector) == config.EMBEDDING_DIMENSION
    assert all(len(vector) == config.EMBEDDING_DIMENSION for vector in document_vectors)
    assert indexed_engine.vector_store is not None
    assert indexed_engine.vector_store.index.d == config.EMBEDDING_DIMENSION
    assert indexed_engine.vector_store.index.ntotal == len(KNOWLEDGE_BASE)
    assert indexed_engine.chunk_count == len(KNOWLEDGE_BASE)


@pytest.mark.parametrize(("question", "expected_chunk_id"), RETRIEVAL_CASES)
def test_faiss_retrieval(indexed_engine: RAGEngine, question: str, expected_chunk_id: int) -> None:
    assert indexed_engine.vector_store is not None
    top_document, _ = indexed_engine.vector_store.similarity_search_with_score(question, k=1)[0]
    matches = indexed_engine.retrieve(question)
    scores = [score for _, score in matches]

    assert top_document.metadata["chunk_id"] == expected_chunk_id
    assert matches[0][0].metadata["chunk_id"] == expected_chunk_id
    assert matches[0][0].page_content == KNOWLEDGE_BASE[expected_chunk_id][0]
    assert len(matches) <= config.TOP_K
    assert all(score >= config.MIN_RELEVANCE_SCORE for score in scores)


def test_keyword_index_ranks_exact_identifiers() -> None:
    index = KeywordIndex(
        [
            "Table 90 Associative container requirements: a_uniq.insert(t) is logarithmic.",
            "Table 91 Unordered associative container requirements: b.find(k) is average case O(1).",
            "std::vector::push_back runs in amortized constant time.",
        ]
    )

    assert index.rank("complexity of find in Table 91", limit=3)[0] == 1
    assert index.rank("a_uniq insert", limit=3)[0] == 0
    assert index.rank("chocolate cake recipe", limit=3) == []
    assert KeywordIndex(["...", "!!"]).rank("anything", limit=3) == []


def test_fallback_mechanism(indexed_engine: RAGEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(**_: Any) -> None:
        raise AssertionError("The LLM must not be called without an API key.")

    monkeypatch.setattr(rag_engine, "ChatGroq", fail_if_called)

    for api_key in (None, "", "   "):
        result = indexed_engine.query(UNORDERED_MAP_QUESTION, api_key=api_key)

        assert result["answer"].startswith(FALLBACK_NOTICE)
        assert "average constant time complexity" in result["answer"]
        assert "page 12" in result["answer"]
        assert 0 < len(result["sources"]) <= config.TOP_K

        top_source = result["sources"][0]
        assert {"page_number", "chunk_id", "snippet"} <= set(top_source)
        assert top_source["chunk_id"] == 0
        assert top_source["page_number"] == 12
        assert top_source["snippet"] == KNOWLEDGE_BASE[0][0]


def test_groq_answer_synthesized_from_retrieved_context(
    indexed_engine: RAGEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm_settings: List[Dict[str, Any]] = []
    prompts: List[List[Any]] = []
    retrieve_calls: List[Tuple[str, str]] = []
    expansion = "std::unordered_map lookup has average case O(1) and worst case O(n) complexity."
    expected_answer = "Lookups in std::unordered_map take average constant time [page 12]."

    def respond(prompt_value: PromptValue) -> AIMessage:
        messages = prompt_value.to_messages()
        prompts.append(messages)
        return AIMessage(content=f"  {expansion if is_expansion_prompt(messages) else expected_answer}\n")

    def fake_groq(**kwargs: Any) -> RunnableLambda:
        llm_settings.append(kwargs)
        return RunnableLambda(respond)

    original_retrieve = indexed_engine.retrieve

    def spy_retrieve(question: str, expansion_text: str = "") -> List[Tuple[Document, float]]:
        retrieve_calls.append((question, expansion_text))
        return original_retrieve(question, expansion_text)

    monkeypatch.setattr(rag_engine, "ChatGroq", fake_groq)
    monkeypatch.setattr(indexed_engine, "retrieve", spy_retrieve)

    result = indexed_engine.query(UNORDERED_MAP_QUESTION, api_key="test-groq-key")
    expansion_prompt, answer_prompt = prompts
    system_message, human_message = answer_prompt

    assert result["answer"] == expected_answer
    assert result["sources"][0]["chunk_id"] == 0
    assert all(settings["model"] == config.DEFAULT_LLM_MODEL for settings in llm_settings)
    assert all(settings["groq_api_key"] == "test-groq-key" for settings in llm_settings)
    assert is_expansion_prompt(expansion_prompt) and expansion_prompt[1].content == UNORDERED_MAP_QUESTION
    assert retrieve_calls == [(UNORDERED_MAP_QUESTION, expansion)]
    assert all(source["snippet"] in system_message.content for source in result["sources"])
    assert "page 12" in system_message.content
    assert human_message.content == UNORDERED_MAP_QUESTION


def test_llm_context_includes_neighbors_from_the_same_page(
    embeddings: Embeddings, monkeypatch: pytest.MonkeyPatch
) -> None:
    table_rows = [
        "Table 91 lists the unordered associative container requirements. The expression b.find(k) returns an "
        "iterator pointing to an element with key equivalent to k, or b.end() if no such element exists. The "
        "container type X must also provide the member types key_type, hasher, and key_equal described above.",
        "Complexity of b.find(k): average case O(1), worst case O(b.size()). The expression b.count(k) returns "
        "the number of elements with key equivalent to k, and its complexity is average case O(b.count(k)) and "
        "worst case O(b.size()). Both expressions are also available on const containers.",
        "The expression b.rehash(n) rehashes the container so that it has at least n buckets. Rehashing "
        "invalidates iterators, changes the ordering between elements, and changes which buckets the elements "
        "appear in, but it does not invalidate pointers or references to the stored elements.",
    ]
    other_page = "Course CS 101 grading policy: the final exam is worth 40 percent of the course grade."
    engine = RAGEngine(embeddings=embeddings, top_k=1, context_neighbors=1, query_expansion=False)
    engine.build_vector_store(
        engine.chunk_text(
            [
                Document(page_content="\n".join(table_rows), metadata={"source": "standard.pdf", "page_number": 880}),
                Document(page_content=other_page, metadata={"source": "standard.pdf", "page_number": 881}),
            ]
        )
    )
    prompts: List[List[Any]] = []

    def respond(prompt_value: PromptValue) -> AIMessage:
        prompts.append(prompt_value.to_messages())
        return AIMessage(content="Average case O(1), worst case O(b.size()) [page 880].")

    monkeypatch.setattr(rag_engine, "ChatGroq", lambda **_: RunnableLambda(respond))

    result = engine.query("What is the average case complexity of b.find(k)?", api_key="test-groq-key")
    context = prompts[0][0].content

    assert engine.chunk_count == 4
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page_number"] == 880
    assert table_rows[1] in context
    assert sum(row in context for row in table_rows) >= 2
    assert other_page not in context
    assert "page 880" in context


def test_answer_citations_and_math_are_tidied(indexed_engine: RAGEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_answer = "Lookups take \\(O(1)\\) on average and \\[O(n)\\] in the worst case 【page 12】."

    def respond(prompt_value: PromptValue) -> AIMessage:
        return AIMessage(content="" if is_expansion_prompt(prompt_value.to_messages()) else raw_answer)

    monkeypatch.setattr(rag_engine, "ChatGroq", lambda **_: RunnableLambda(respond))

    result = indexed_engine.query(UNORDERED_MAP_QUESTION, api_key="test-groq-key")

    assert result["answer"] == "Lookups take $O(1)$ on average and $$O(n)$$ in the worst case [page 12]."


def test_failed_query_expansion_still_answers(indexed_engine: RAGEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    expected_answer = "Lookups in std::unordered_map take average constant time [page 12]."

    def respond(prompt_value: PromptValue) -> AIMessage:
        if is_expansion_prompt(prompt_value.to_messages()):
            raise TimeoutError("expansion timed out")
        return AIMessage(content=expected_answer)

    monkeypatch.setattr(rag_engine, "ChatGroq", lambda **_: RunnableLambda(respond))

    result = indexed_engine.query(UNORDERED_MAP_QUESTION, api_key="test-groq-key")

    assert result["answer"] == expected_answer
    assert result["sources"][0]["chunk_id"] == 0


def test_llm_failure_degrades_to_fallback(indexed_engine: RAGEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreachable_groq(**_: Any) -> None:
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(rag_engine, "ChatGroq", unreachable_groq)

    result = indexed_engine.query(UNORDERED_MAP_QUESTION, api_key="test-groq-key")

    assert result["answer"].startswith("The language model request failed (ConnectionError)")
    assert "average constant time complexity" in result["answer"]
    assert result["sources"][0]["chunk_id"] == 0


def test_query_handles_empty_question(indexed_engine: RAGEngine) -> None:
    for question in ("", "   ", "\n\t"):
        assert indexed_engine.query(question) == {"answer": EMPTY_QUESTION_MESSAGE, "sources": []}


def test_query_without_index_is_graceful() -> None:
    engine = RAGEngine()

    assert not engine.is_ready
    assert engine.query("What is the complexity of std::sort?") == {"answer": NO_INDEX_MESSAGE, "sources": []}


def test_query_with_no_relevant_match(indexed_engine: RAGEngine) -> None:
    result = indexed_engine.query("What is a good recipe for chocolate cake?")

    assert result == {"answer": NO_MATCH_MESSAGE, "sources": []}
