"""Comprehensive unit and integration test suite for the RAG pipeline."""

import os
from pathlib import Path
import unittest
import rag

def test_mock_embeddings():
    """Verify MockEmbeddings produce valid normalized float vectors of expected dimension."""
    embedder = rag.MockEmbeddings(dimension=512)
    doc_vecs = embedder.embed_documents(["Attention is all you need.", "Recurrent neural network."])
    query_vec = embedder.embed_query("Attention layers")

    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 512
    assert len(query_vec) == 512

    # Verify L2 normalization
    norm = sum(x * x for x in query_vec) ** 0.5
    assert abs(norm - 1.0) < 1e-4

def test_load_and_split():
    """Verify PDF loading and document splitting."""
    assert rag.SAMPLE_PDF.exists()
    docs = rag.load_pdf(rag.SAMPLE_PDF)
    assert len(docs) >= 1

    chunks = rag.split_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= len(docs)
    for c in chunks:
        assert len(c.page_content) <= 600
        assert "page" in c.metadata

def test_vectorstore_and_retriever():
    """Verify Chroma vector store creation and retriever."""
    assistant = rag.create_assistant(
        pdf_path=rag.SAMPLE_PDF,
        chunk_size=1000,
        chunk_overlap=150,
        k=3,
        is_mock=True,
    )
    assert assistant is not None
    stats = assistant.get_stats()
    assert stats["chunk_count"] > 0
    assert stats["is_mock"] is True

    # Test retrieval
    standalone, docs = assistant.retrieve("What are the limitations?", [])
    assert len(docs) > 0
    assert standalone == "What are the limitations?"

def test_standalone_question_rewriting():
    """Verify follow-up question resolution in mock mode."""
    assistant = rag.create_assistant(is_mock=True)
    history = [
        {"role": "user", "content": "What is self-attention?"},
        {"role": "assistant", "content": "Self-attention is a weighted combination mechanism."},
    ]
    rewritten = assistant.standalone_question("What are its limitations?", history)
    assert "self" in rewritten.lower() or "attention" in rewritten.lower()

def test_qa_accuracy():
    """Verify factual question answering across key topics."""
    assistant = rag.create_assistant(is_mock=True)

    # 1. Main idea
    ans1, _ = assistant.ask("What is the main idea of this paper?")
    assert "Overview" in ans1 or "Key Takeaway" in ans1
    assert "p." in ans1

    # 2. Limitations
    ans2, _ = assistant.ask("What are the limitations?")
    assert "quadratically" in ans2.lower() or "expensive" in ans2.lower() or "untested" in ans2.lower()
    assert "p. 2" in ans2

    # 3. Results
    ans3, _ = assistant.ask("What were the results?")
    assert "bleu" in ans3.lower()
    assert "28.4" in ans3 or "41.0" in ans3
    assert "p. 2" in ans3

    # 4. Training
    ans4, _ = assistant.ask("What optimizer and hardware were used for training?")
    assert "adam" in ans4.lower() or "gpu" in ans4.lower()
    assert "p. 2" in ans4

if __name__ == "__main__":
    test_mock_embeddings()
    test_load_and_split()
    test_vectorstore_and_retriever()
    test_standalone_question_rewriting()
    test_qa_accuracy()
    print("All tests passed successfully!")
