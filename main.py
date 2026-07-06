"""
main.py — Entry point for the repo-assistant.

Full pipeline:
  RepoCloner → RepoWalker → FileIndexer → FileEmbedder → VectorStore → RepoBot (RAG)
"""

import sys

from repo_assistant.traversal import RepoCloner, RepoWalker
from repo_assistant.indexer import FileIndexer
from repo_assistant.analyzer import ASTPythonParser, CodeGraphAnalyzer
from repo_assistant.embedder import FileEmbedder, SentenceTransformerBackend
from repo_assistant.store import VectorStore
from repo_assistant.chat import RepoBot, OllamaBackend, CrossEncoderReranker


def main():
    print("Welcome to AI Repo Assistant!")
    print("Enter a GitHub URL to analyze (or press Enter to use default: pallets/flask):")
    
    url = input("> ").strip()
    if not url:
        url = "https://github.com/pallets/flask"

    print(f"\n[1/5] Cloning/Loading repository: {url} ...")
    cloner = RepoCloner()
    repo_path = cloner.clone(url)

    print(f"[2/5] Finding files...")
    walker = RepoWalker()
    file_paths = walker.walk(repo_path)

    print(f"[3/5] Indexing files...")
    indexer = FileIndexer()
    indexed_files = indexer.index(file_paths, repo_root=repo_path)
    
    # Take a sample to keep the boot time fast for the demonstration
    print(f"      (Using first 50 files for speed in this demo)")
    indexed_files = indexed_files[:50]

    print(f"[3.5/5] Extracting AST Code Graph...")
    parser = ASTPythonParser()
    analyzer = CodeGraphAnalyzer(parser)
    semantic_files = analyzer.analyze(indexed_files)
    
    if semantic_files:
        print(f"        -> Extracted {len(semantic_files)} semantic class/function signatures.")
        # DO NOT append to indexed_files. They are treated as a separate collection logically.
        
    skeleton = analyzer.generate_skeleton()

    print(f"[4/5] Generating local embeddings (SentenceTransformers)...")
    embed_backend = SentenceTransformerBackend()
    embedder = FileEmbedder(embed_backend)
    
    embedded_source_files = embedder.embed(indexed_files)
    embedded_semantic_files = embedder.embed(semantic_files) if semantic_files else []

    print(f"[5/5] Building ChromaDB vector store...")
    # Using ephemeral memory store for the CLI session
    store = VectorStore(embed_backend, persist_directory=None)
    
    # Store source files in the "source" namespace
    store.build(embedded_source_files, namespace="source")
    
    # Store semantic files in the "semantic" namespace
    if embedded_semantic_files:
        store.build(embedded_semantic_files, namespace="semantic")

    print("\n✅ Repository loaded and indexed!")
    print("Initializing Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Connecting to local Ollama (model: llama3)...")
    
    chat_backend = OllamaBackend(model_name="llama3")
    bot = RepoBot(store, chat_backend, reranker=reranker, skeleton=skeleton)

    print("\n" + "="*50)
    print("Chat Session Started.")
    print("Commands:")
    print("  /show <symbol>  - Jump directly to a class/function definition.")
    print("  /refs <symbol>  - Find all references to a symbol across the codebase.")
    print("  quit / exit     - End the session.")
    print("="*50 + "\n")

    while True:
        try:
            question = input("\n👤 You: ").strip()
            if not question:
                continue
            if question.lower() in ['quit', 'exit']:
                break
                
            if question.lower().startswith("/show "):
                symbol = question[6:].strip()
                result = analyzer.find_symbol(symbol)
                print(f"\n🔍 {result}")
                continue

            if question.lower().startswith("/refs "):
                symbol = question[6:].strip()
                result = analyzer.format_references(symbol)
                print(f"\n🔍 {result}")
                continue

            print("\n🤖 Assistant: ", end="", flush=True)
            sources, stream = bot.ask_stream(question, top_k=3)
            
            for chunk in stream:
                print(chunk, end="", flush=True)
            print()
            
            if sources:
                print("\nSources:")
                for i, src in enumerate(sources, 1):
                    print(f"  [{i}] {src.indexed_file.relative_path} (score: {src.score:.2f})")
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
