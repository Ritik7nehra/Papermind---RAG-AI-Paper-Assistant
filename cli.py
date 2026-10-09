"""Terminal version: chat with a research paper (PDF).

Usage:
    python cli.py                              # uses sample paper (data/paper.pdf)
    python cli.py path/to/paper.pdf            # chat with any custom PDF
    python cli.py --mock                       # force offline simulated mode
    python cli.py --openai --model gpt-4o-mini # use OpenAI API
"""

import argparse
import sys
from pathlib import Path

import rag

# Safe stdout reconfiguration for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Simple ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
DIM = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}==================================================================
           AI RESEARCH PAPER ASSISTANT (CLI)
=================================================================={RESET}
{DIM}Built with LangChain, Chroma Vector Store, and Retrieval Augmented Generation.{RESET}
Type your question, or use commands:
  {YELLOW}/sources{RESET}  - Display the full retrieved chunks from previous turn
  {YELLOW}/stats{RESET}    - Show active document and vector store statistics
  {YELLOW}/clear{RESET}    - Clear conversation history
  {YELLOW}/help{RESET}     - Show available commands
  {YELLOW}quit{RESET}      - Exit the CLI
"""
    print(banner)


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with research papers in terminal.")
    parser.add_argument("pdf", nargs="?", default=str(rag.SAMPLE_PDF), help="Path to PDF paper")
    parser.add_argument("--mock", action="store_true", help="Force simulated offline mode")
    parser.add_argument("--openai", action="store_true", help="Force OpenAI API mode")
    parser.add_argument("--model", default=None, help="OpenAI chat model (e.g. gpt-4o-mini, gpt-4o)")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks retrieved per question")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters")
    return parser.parse_args()


def main():
    args = parse_args()
    pdf_path = Path(args.pdf)

    if not pdf_path.exists():
        sys.exit(f"{YELLOW}Error:{RESET} PDF not found at '{pdf_path}'")

    print_banner()

    # Determine execution mode
    if args.mock:
        use_mock = True
    elif args.openai:
        use_mock = False
    else:
        use_mock = rag.is_mock_mode()

    if use_mock:
        print(f"[{YELLOW}Mode:{RESET} Simulated / Mock LLM (Offline & Free)]")
        print(f"{DIM}Note: Answers and embeddings run locally without API charges.{RESET}")
        print(f"{DIM}To switch to OpenAI, set OPENAI_API_KEY in .env or pass --openai.{RESET}\n")
    else:
        if not rag.api_key_is_set():
            sys.exit(
                f"\n{YELLOW}No OpenAI API key found.{RESET}\n"
                "Please put OPENAI_API_KEY=sk-... in your .env file or run with --mock for offline mode.\n"
            )
        model_name = args.model or rag.CHAT_MODEL
        print(f"[{GREEN}Mode:{RESET} OpenAI API ({model_name})]\n")

    print(f"Indexing {BOLD}{pdf_path.name}{RESET} (cached in chroma_db/)...")
    try:
        chat = rag.create_assistant(
            pdf_path=pdf_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            k=args.k,
            is_mock=use_mock,
            model_name=args.model,
        )
    except Exception as exc:
        sys.exit(f"Error initializing assistant: {exc}")

    stats = chat.get_stats()
    print(f"{DIM}Ready! {stats['chunk_count']} chunks indexed in collection '{stats['collection_name']}'.{RESET}")

    history = []
    last_docs = []
    last_query = ""

    while True:
        try:
            prompt_str = f"\n{GREEN}{BOLD}You:{RESET} "
            question = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}Goodbye!{RESET}")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print(f"\n{CYAN}Goodbye!{RESET}")
            break

        if question.lower() == "/clear":
            history = []
            last_docs = []
            print(f"{YELLOW}Conversation memory cleared.{RESET}")
            continue

        if question.lower() == "/stats":
            st = chat.get_stats()
            print(f"\n{CYAN}Document & Index Stats:{RESET}")
            print(f"  • File: {st['pdf_name']}")
            print(f"  • Total Chunks: {st['chunk_count']}")
            print(f"  • Collection: {st['collection_name']}")
            print(f"  • Model: {st['model_name']}")
            print(f"  • Simulated Mode: {st['is_mock']}")
            continue

        if question.lower() == "/sources":
            if not last_docs:
                print(f"{DIM}No retrieved chunks recorded yet. Ask a question first!{RESET}")
            else:
                print(f"\n{CYAN}--- Retrieved Chunks for '{last_query}' ---{RESET}")
                for idx, doc in enumerate(last_docs, 1):
                    p_num = doc.metadata.get("page", 0) + 1
                    print(f"\n{BOLD}[Chunk #{idx} - Page {p_num}]{RESET}")
                    print(f"{DIM}{doc.page_content.strip()}{RESET}")
                print(f"{CYAN}---------------------------------------------{RESET}")
            continue

        if question.lower() == "/help":
            print(f"\n{CYAN}Available Commands:{RESET}")
            print("  /sources  - View full text of chunks retrieved for the previous turn")
            print("  /stats    - Display collection and document stats")
            print("  /clear    - Reset the conversation history")
            print("  /help     - Display this command reference")
            print("  quit      - Exit the assistant")
            continue

        try:
            searched, docs = chat.retrieve(question, history)
            last_docs = docs
            last_query = searched

            print(f"\n{CYAN}{BOLD}Assistant:{RESET} ", end="", flush=True)
            answer = ""
            for piece in chat.stream_answer(question, history, docs):
                try:
                    print(piece, end="", flush=True)
                except UnicodeEncodeError:
                    print(piece.encode("ascii", errors="replace").decode("ascii"), end="", flush=True)
                answer += piece
            print()

            pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
            p_str = ", ".join(map(str, pages)) if pages else "None"
            print(f"{DIM}[Sources: Pages {p_str} | Retrieved {len(docs)} chunks]{RESET}")

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

        except Exception as err:
            print(f"\n{YELLOW}[Error generating answer: {err}]{RESET}")


if __name__ == "__main__":
    main()
