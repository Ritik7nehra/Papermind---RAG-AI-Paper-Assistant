"""RAG pipeline for the AI Research Paper Assistant.

Indexing (once per PDF):
    PDF -> load -> split -> embeddings -> Chroma

Answering (every question):
    question + chat history -> standalone question
        -> retriever -> relevant chunks -> prompt -> LLM -> answer
"""

import hashlib
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_PDF = DATA_DIR / "paper.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"
UPLOADS_DIR = DATA_DIR / "uploads"

CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# How many past messages are sent back to the model with each question.
MAX_HISTORY_MESSAGES = 10

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and the user's latest question, rewrite "
            "the latest question so it can be understood on its own (replace "
            "pronouns like 'it', 'they', 'its', 'their' with what they refer to). "
            "Do NOT answer it. If it is already self-contained, return it unchanged. "
            "Output only the rewritten question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert AI research assistant helping the user thoroughly understand a research paper. "
            "Answer using ONLY the provided context excerpts from the paper. "
            "Ground every claim strictly in the context. If the context does not contain the answer, "
            "explicitly state that the provided excerpts do not cover it rather than hallucinating. "
            "Cite page numbers like (p. 2) for key findings, numbers, or architecture details. "
            "Follow any requested format, such as 'in 5 points' or structured sections.\n\n"
            "Context excerpts:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


# --- Mock / Offline Embeddings ---------------------------------------------
class MockEmbeddings(Embeddings):
    """Deterministic, high-fidelity hashing vectorizer for offline RAG testing.

    Uses sublinear term frequency, English stopword reduction, and stem-based
    hashing across 512 dimensions for reliable lexical retrieval without an API key.
    """

    STOPWORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "as", "at", "be", "because", "been", "before",
        "being", "below", "between", "both", "but", "by", "can", "did", "do",
        "does", "doing", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers",
        "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
        "is", "it", "its", "itself", "just", "me", "more", "most", "my",
        "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
        "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
        "same", "she", "should", "so", "some", "such", "than", "that", "the",
        "their", "theirs", "them", "themselves", "then", "there", "these",
        "they", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "we", "were", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "you", "your", "yours",
    }

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.model = "mock-embedding-512"

    @staticmethod
    def _stem(word: str) -> str:
        """Lightweight suffix normalization."""
        for suffix in ("tions", "tion", "ing", "ies", "ied", "ed", "es", "s", "ly", "ment", "al"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
        if not words:
            return vec

        counts: dict[str, float] = {}
        for w in words:
            if len(w) <= 1:
                continue
            is_stop = w in self.STOPWORDS
            stem = self._stem(w)
            weight = 0.05 if is_stop else 1.0
            counts[stem] = counts.get(stem, 0.0) + weight

        # Add adjacent bigrams for multi-word phrase matching
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 not in self.STOPWORDS and w2 not in self.STOPWORDS:
                bg = f"{self._stem(w1)}_{self._stem(w2)}"
                counts[bg] = counts.get(bg, 0.0) + 1.5

        for token, w in counts.items():
            tf = 1.0 + math.log(w) if w >= 1.0 else w
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dimension] += tf

        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def is_mock_mode() -> bool:
    """Return True if simulated/mock mode is enabled or no valid API key is present."""
    if os.getenv("ENABLE_MOCK_MODE", "").lower() in ("true", "1", "yes"):
        return True
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return not key or key in ("your_api_key_here", "mock-mode", "mock")


def api_key_is_set() -> bool:
    """Return True if a valid non-placeholder OpenAI key is configured."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key) and key not in ("your_api_key_here", "mock-mode", "mock")


# --- Load ------------------------------------------------------------------
def load_pdf(pdf_path: Union[str, Path] = SAMPLE_PDF) -> List[Document]:
    """Load a PDF into a list of Documents (one per page)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} not found.")
    return PyPDFLoader(str(pdf_path)).load()


# --- Split -----------------------------------------------------------------
def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """Split pages into overlapping chunks that fit comfortably in a prompt."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --- Embeddings + Chroma ---------------------------------------------------
def file_fingerprint(pdf_path: Union[str, Path]) -> str:
    """Short hash of the PDF's bytes, so the same file is only embedded once."""
    return hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()[:16]


def safe_collection_name(
    pdf_path: Union[str, Path],
    chunk_size: int,
    chunk_overlap: int,
    model_identifier: str,
) -> str:
    """Generate a Chroma-compatible collection name (3-63 alphanumeric/dash chars)."""
    fp = file_fingerprint(pdf_path)
    clean_model = re.sub(r"[^a-zA-Z0-9]+", "-", str(model_identifier)).strip("-")[:16]
    name = f"paper-{fp}-{chunk_size}-{chunk_overlap}-{clean_model}".strip("-")
    if len(name) > 60:
        h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:50].rstrip('-')}-{h}"
    return name


def build_vectorstore(
    pdf_path: Union[str, Path] = SAMPLE_PDF,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    persist_dir: Union[str, Path] = CHROMA_DIR,
    embeddings: Optional[Embeddings] = None,
    use_mock: Optional[bool] = None,
) -> Chroma:
    """Embed the PDF's chunks into Chroma and return the vector store.

    Each (PDF contents, chunk settings, embedding model) combination gets its own
    Chroma collection on disk. Re-opening the same paper with the same settings
    reuses the stored vectors instead of paying for embeddings again.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    if use_mock is None:
        use_mock = is_mock_mode() if embeddings is None else False

    if embeddings is None:
        embeddings = MockEmbeddings() if use_mock else OpenAIEmbeddings(model=EMBEDDING_MODEL)

    model_name = getattr(embeddings, "model", "custom")
    collection = safe_collection_name(pdf_path, chunk_size, chunk_overlap, model_name)

    store = Chroma(
        collection_name=collection,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    if not store.get(limit=1)["ids"]:
        chunks = split_documents(load_pdf(pdf_path), chunk_size, chunk_overlap)
        chunks = [c for c in chunks if c.page_content.strip()]
        if not chunks:
            raise ValueError(
                "No text could be extracted from this PDF. It may be a scanned "
                "document (images only); run OCR on it first."
            )
        store.add_documents(chunks)

    return store


# --- Retriever -------------------------------------------------------------
def get_retriever(vectorstore: Chroma, k: int = 4):
    """Return the k chunks most similar to a question."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs: List[Document]) -> str:
    """Join retrieved chunks into one context string, tagged with page numbers."""
    return "\n\n".join(
        f"[page {doc.metadata.get('page', 0) + 1}] {doc.page_content}"
        for doc in docs
    )


# --- Conversation ----------------------------------------------------------
def to_messages(history: list) -> list:
    """Turn [{'role': 'user'|'assistant', 'content': str}] into LangChain messages."""
    messages = []
    for turn in history[-MAX_HISTORY_MESSAGES:]:
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))
    return messages


def _stem(word: str) -> str:
    """Lightweight word stemmer for question matching."""
    for suffix in ("tions", "tion", "ing", "ies", "ied", "ed", "es", "s", "ly", "ment", "al"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def synthesize_mock_answer(question: str, history: list, docs: List[Document]) -> Generator[str, None, None]:
    """Simulate an expert AI research assistant streaming an accurate answer.

    Extracts, cleans, ranks, and synthesizes key information from the retrieved
    document chunks with exact page grounding.
    """
    if not docs:
        msg = "The paper excerpts do not cover this question, as no relevant text was retrieved."
        for word in msg.split():
            yield word + " "
            time.sleep(0.01)
        return

    q_lower = question.lower().strip()
    stopwords = {
        "what", "is", "the", "in", "of", "and", "a", "an", "to", "how", "why",
        "are", "do", "does", "did", "this", "that", "these", "those", "for", "with",
        "on", "about", "can", "you", "tell", "me", "summarize", "explain", "describe",
        "it", "its", "paper", "they", "there", "their", "give", "show",
    }
    raw_keywords = [w for w in re.findall(r"\w+", q_lower) if w not in stopwords and len(w) > 2]
    stemmed_keywords = {_stem(w) for w in raw_keywords}

    # Clean text across chunks and extract sentences tagged with metadata
    all_sentences = []
    seen_sentences = set()

    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        raw_text = doc.page_content

        # Normalize line breaks within paragraphs
        normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", raw_text)
        paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]

        current_section = "General"
        for p in paragraphs:
            # Check if paragraph starts with a section header
            header_match = re.match(r"^(\d+\.?\s+[A-Za-z\s-]+|Abstract|Introduction|Conclusion|Limitations|Results|Model Architecture|Training|Methodology)", p)
            if header_match:
                current_section = header_match.group(1).strip()

            # Split sentences
            sentences = re.split(r"(?<=[.!?])\s+", p)
            for s in sentences:
                s_clean = s.strip()
                # Skip placeholder or boilerplate text
                if "replace it with a real paper" in s_clean.lower() or "this file is a placeholder" in s_clean.lower():
                    continue
                # Strip leading section header if attached to sentence
                s_clean = re.sub(
                    r"^(\d+\.?\s+[A-Za-z\s-]+|Abstract|Introduction|Conclusion|Limitations|Results|Model Architecture|Training|Methodology)\s*[:.-]?\s*",
                    "",
                    s_clean,
                    flags=re.IGNORECASE,
                ).strip()
                if len(s_clean) >= 20 and s_clean not in seen_sentences:
                    seen_sentences.add(s_clean)
                    words_in_s = re.findall(r"\w+", s_clean.lower())
                    stems_in_s = {_stem(w) for w in words_in_s}

                    # Compute score
                    kw_matches = sum(1 for st in stemmed_keywords if st in stems_in_s)
                    sec_match = 1 if any(_stem(kw) in current_section.lower() for kw in raw_keywords) else 0

                    score = kw_matches * 2.0 + sec_match * 3.0
                    all_sentences.append({
                        "text": s_clean,
                        "page": page,
                        "section": current_section,
                        "score": score,
                        "kw_matches": kw_matches,
                    })

    # Sort sentences by relevance
    all_sentences.sort(key=lambda x: x["score"], reverse=True)

    # Intent detection
    is_overview = any(k in q_lower for k in ["main idea", "overview", "what is this paper", "about", "summary", "contribution"])
    is_method = any(k in q_lower for k in ["5 points", "points", "methodology", "method", "architecture", "mechanism", "how does it work", "steps"])
    is_limitations = any(k in q_lower for k in ["limitation", "limitations", "weakness", "drawback", "challenges", "constraints", "future work"])
    is_results = any(k in q_lower for k in ["result", "results", "bleu", "score", "performance", "benchmark", "accuracy", "findings"])
    is_training = any(k in q_lower for k in ["training", "train", "optimizer", "learning rate", "adam", "dropout", "gpu", "hardware", "hyperparameters"])

    response_parts = []

    if is_overview:
        response_parts.append("### Overview & Core Innovation\n\n")
        response_parts.append(
            f"Based on the paper excerpts (e.g., p. {docs[0].metadata.get('page', 0) + 1}), here are the central concepts:\n\n"
        )
        overview_sents = [
            s for s in all_sentences
            if any(sec in s["section"].lower() for sec in ["abstract", "introduction", "conclusion", "model architecture"])
            or s["score"] > 0
        ][:4]
        if not overview_sents:
            overview_sents = all_sentences[:4]
        for s in overview_sents:
            response_parts.append(f"- **Key Takeaway:** {s['text']} *(p. {s['page']})*\n")

    elif is_method:
        response_parts.append("### Methodology & Architecture in 5 Points\n\n")
        method_sents = [
            s for s in all_sentences
            if any(term in s["section"].lower() for term in ["architecture", "attention", "model", "method", "training"])
            or s["score"] > 0
        ]
        if len(method_sents) < 5:
            # Fall back to top general sentences
            for s in all_sentences:
                if s not in method_sents:
                    method_sents.append(s)
                if len(method_sents) >= 5:
                    break

        for idx, s in enumerate(method_sents[:5], 1):
            response_parts.append(f"{idx}. **{s['section']}:** {s['text']} *(p. {s['page']})*\n")

    elif is_limitations:
        response_parts.append("### Scope, Limitations & Constraints\n\n")
        limit_sents = [
            s for s in all_sentences
            if "limitation" in s["section"].lower()
            or any(w in s["text"].lower() for w in ["limitation", "quadratic", "expensive", "cost", "untested", "constraint", "drawback"])
        ]
        if limit_sents:
            for s in limit_sents:
                response_parts.append(f"- **Identified Constraint:** {s['text']} *(p. {s['page']})*\n")
        else:
            response_parts.append(
                f"- The retrieved excerpts from p. {docs[0].metadata.get('page', 0) + 1} do not explicitly list formal negative constraints. "
                "The study emphasizes its self-attention design and primary benchmark.\n"
            )

    elif is_results:
        response_parts.append("### Empirical Results & Benchmarks\n\n")
        res_sents = [
            s for s in all_sentences
            if "result" in s["section"].lower()
            or any(w in s["text"].lower() for w in ["bleu", "score", "improving", "benchmark", "accuracy", "test set", "points"])
        ]
        if res_sents:
            for s in res_sents:
                response_parts.append(f"- **Evaluation Metric:** {s['text']} *(p. {s['page']})*\n")
        else:
            response_parts.append("No specific quantitative benchmark was identified in the retrieved sections.\n")

    elif is_training:
        response_parts.append("### Training Setup & Hyperparameters\n\n")
        train_sents = [
            s for s in all_sentences
            if "train" in s["section"].lower()
            or any(w in s["text"].lower() for w in ["adam", "optimizer", "gpu", "gpus", "hours", "dropout", "learning-rate", "schedule", "dataset"])
        ]
        if train_sents:
            for s in train_sents:
                response_parts.append(f"- **Setup Detail:** {s['text']} *(p. {s['page']})*\n")
        else:
            response_parts.append("Training hyperparameters were not found in the current retrieved chunks.\n")

    else:
        # General query: check top matching sentences
        top_matches = [s for s in all_sentences if s["score"] > 0]
        if top_matches:
            response_parts.append(f"Based on the retrieved excerpts regarding **{question}**:\n\n")
            for s in top_matches[:4]:
                response_parts.append(f"- {s['text']} *(p. {s['page']})*\n")
        else:
            first_p = docs[0].metadata.get("page", 0) + 1
            snippet = docs[0].page_content.strip().split("\n")[0][:180]
            response_parts.append(
                f"The paper excerpts do not directly contain specific details for: *\"{question}\"*.\n\n"
                f"The closest retrieved section on **Page {first_p}** discusses:\n\n"
                f"> \"{snippet}...\"\n\n"
                f"Note: Try increasing the number of retrieved chunks (k) in the settings or rephrasing with specific keywords from the paper."
            )

    response_parts.append(
        "\n\n*(Running in [Simulated / Mock LLM Mode]. Responses are synthesized locally from retrieved PDF chunks. "
        "Set `OPENAI_API_KEY` in `.env` or in the sidebar to enable full GPT reasoning.)*"
    )

    full_text = "".join(response_parts)
    tokens = re.findall(r"\S+|\n", full_text)
    for token in tokens:
        if token == "\n":
            yield "\n"
        else:
            yield token + " "
            time.sleep(0.01)


class PaperChat:
    """Chat with one paper: retrieval + conversation memory.

    The chat history lives with the caller (Streamlit session state, a list in
    the terminal app), so one PaperChat can serve any number of conversations.
    """

    def __init__(
        self,
        retriever,
        llm=None,
        is_mock: Optional[bool] = None,
        model_name: Optional[str] = None,
        vectorstore: Optional[Chroma] = None,
        pdf_path: Optional[Union[str, Path]] = None,
    ):
        self.is_mock = is_mock_mode() if is_mock is None else is_mock
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.pdf_path = Path(pdf_path) if pdf_path else None
        self.model_name = model_name or CHAT_MODEL

        if not self.is_mock:
            self.llm = llm or ChatOpenAI(model=self.model_name)
            self.condense_chain = CONDENSE_PROMPT | self.llm | StrOutputParser()
            self.answer_chain = ANSWER_PROMPT | self.llm | StrOutputParser()
        else:
            self.llm = None
            self.condense_chain = None
            self.answer_chain = None

    def standalone_question(self, question: str, history: list) -> str:
        """Rewrite a follow-up ('Why did they do that?') into a full standalone question."""
        if not history:
            return question

        if self.is_mock:
            pronouns = ["it", "they", "its", "their", "this", "that", "these", "those", "the model", "the paper"]
            q_lower = question.lower()
            if any(re.search(r"\b" + re.escape(p) + r"\b", q_lower) for p in pronouns):
                last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
                if last_user:
                    subject_words = [w for w in re.findall(r"\w+", last_user) if len(w) > 3]
                    if subject_words:
                        return f"{question} ({' '.join(subject_words[:3])})"
            return question

        try:
            rewritten = self.condense_chain.invoke(
                {"question": question, "chat_history": to_messages(history)}
            )
            return rewritten.strip() or question
        except Exception:
            return question

    def retrieve(self, question: str, history: list) -> Tuple[str, List[Document]]:
        """Return (standalone_question, relevant_chunks)."""
        standalone = self.standalone_question(question, history)
        return standalone, self.retriever.invoke(standalone)

    def stream_answer(self, question: str, history: list, docs: List[Document]) -> Generator[str, None, None]:
        """Yield the answer piece by piece from the retrieved chunks."""
        if self.is_mock:
            yield from synthesize_mock_answer(question, history, docs)
        else:
            yield from self.answer_chain.stream(
                {
                    "question": question,
                    "chat_history": to_messages(history),
                    "context": format_docs(docs),
                }
            )

    def ask(self, question: str, history=()) -> Tuple[str, List[Document]]:
        """Non-streaming helper: returns (answer, source_chunks)."""
        history = list(history)
        _, docs = self.retrieve(question, history)
        return "".join(self.stream_answer(question, history, docs)), docs

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics of the indexed document."""
        chunk_count = 0
        collection_name = "N/A"
        if self.vectorstore is not None:
            try:
                chunk_count = self.vectorstore._collection.count()
                collection_name = self.vectorstore._collection.name
            except Exception:
                pass
        return {
            "pdf_name": self.pdf_path.name if self.pdf_path else "paper.pdf",
            "chunk_count": chunk_count,
            "collection_name": collection_name,
            "model_name": self.model_name,
            "is_mock": self.is_mock,
        }


def create_assistant(
    pdf_path: Union[str, Path] = SAMPLE_PDF,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    k: int = 4,
    embeddings: Optional[Embeddings] = None,
    llm=None,
    is_mock: Optional[bool] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> PaperChat:
    """Build the whole pipeline for one PDF and return a PaperChat."""
    use_mock = is_mock_mode() if is_mock is None else is_mock
    effective_model = model_name or CHAT_MODEL

    if not use_mock:
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
            llm = llm or ChatOpenAI(model=effective_model, temperature=temperature, api_key=api_key)
        else:
            embeddings = embeddings or OpenAIEmbeddings(model=EMBEDDING_MODEL)
            llm = llm or ChatOpenAI(model=effective_model, temperature=temperature)

    store = build_vectorstore(
        pdf_path,
        chunk_size,
        chunk_overlap,
        embeddings=embeddings,
        use_mock=use_mock,
    )

    return PaperChat(
        get_retriever(store, k=k),
        llm=llm,
        is_mock=use_mock,
        model_name=effective_model,
        vectorstore=store,
        pdf_path=pdf_path,
    )
