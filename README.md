# 🚀 AI Repository Assistant

> An intelligent local-first codebase assistant that understands software repositories using Retrieval-Augmented Generation (RAG), AST-based code analysis, semantic search, and local LLMs.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Embeddings](https://img.shields.io/badge/Embeddings-SentenceTransformers-orange)
![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![LLM](https://img.shields.io/badge/LLM-Ollama-red)

---

# 📖 Overview

Understanding large repositories is one of the biggest challenges developers face.

Instead of manually opening dozens of files, searching symbols, and tracing function calls, **AI Repository Assistant** enables developers to interact with a repository using natural language.

It combines:

- 🔍 Semantic Search
- 🧠 Retrieval-Augmented Generation (RAG)
- 🌳 Abstract Syntax Tree (AST) Analysis
- 📚 Code Graph Generation
- 🤖 Local LLMs via Ollama
- ⚡ Cross-Encoder Re-ranking

Everything runs **locally**, keeping your source code private.

---

# ✨ Features

## Repository Processing

- Clone any GitHub repository
- Recursive repository traversal
- Intelligent file filtering
- Source code indexing

---

## Semantic Search

- Character-based smart chunking
- Local SentenceTransformer embeddings
- ChromaDB vector storage
- Semantic similarity search

---

## AI Question Answering

- Retrieval-Augmented Generation (RAG)
- Local Ollama integration
- Context-aware prompting
- Streaming responses
- Cross-Encoder re-ranking for higher accuracy

---

## AST Code Intelligence

- Python AST parsing
- Extract classes
- Extract functions
- Generate semantic signatures
- Repository code skeleton generation

---

## Symbol Navigation

### Go To Definition

```
/show Flask
```

Example output

```
Found Class Flask

Location:
flask/app.py

Methods:
run()
test_client()
route()
...
```

---

### Find References

```
/refs Blueprint
```

Example output

```
Blueprint

Referenced in:

flask/app.py
flask/views.py
flask/helpers.py
...
```

---

# 🏗 Architecture

```
                    GitHub Repository
                           │
                           ▼
                    Repository Cloner
                           │
                           ▼
                    Repository Walker
                           │
                           ▼
                     File Indexer
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
   AST Code Analyzer                 File Chunker
          │                                 │
          ▼                                 ▼
 Semantic Documents                Source Documents
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
             SentenceTransformer
          (all-MiniLM-L6-v2)
                         │
                         ▼
                  ChromaDB Vector DB
                         │
                         ▼
              Cross Encoder Reranker
                         │
                         ▼
                    Ollama (Llama3)
                         │
                         ▼
                  Natural Language Answer
```

---

# 🧠 Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python 3.13 |
| Vector Database | ChromaDB |
| Embeddings | SentenceTransformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Ollama |
| Parser | Python AST |
| Testing | PyTest |

---

# 📂 Project Structure

```
repo-assistant
│
├── repo_assistant/
│   ├── analyzer/
│   ├── chat/
│   ├── embedder/
│   ├── indexer/
│   ├── store/
│   ├── traversal/
│   └── ...
│
├── tests/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/ruthwik1512/repo-assistant.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Install Ollama

https://ollama.com

Download a model

```bash
ollama pull llama3
```

Run

```bash
python main.py
```

---

# 💻 Usage

Launch the assistant

```bash
python main.py
```

Example questions

```
Explain the architecture.

How does authentication work?

Where is Flask initialized?

How are routes registered?

Explain request lifecycle.
```

Navigation commands

```
/show Flask

/refs Blueprint
```

---

# 📈 Current Capabilities

- ✅ Repository cloning
- ✅ Repository traversal
- ✅ Source code indexing
- ✅ Smart chunking
- ✅ Local embeddings
- ✅ ChromaDB integration
- ✅ Retrieval-Augmented Generation
- ✅ Local LLM inference
- ✅ Cross-Encoder re-ranking
- ✅ AST parsing
- ✅ Semantic code graph
- ✅ Code skeleton generation
- ✅ Go To Definition (`/show`)
- ✅ Find References (`/refs`)

---

# 🛣 Roadmap

## Version 1.1

- [ ] Call Graph Generation
- [ ] Caller/Callee Navigation

---

## Version 1.2

- [ ] Tree-sitter
- [ ] Multi-language support
- [ ] Java
- [ ] C++
- [ ] JavaScript
- [ ] TypeScript
- [ ] Go

---

## Version 1.3

- [ ] Dependency Graph
- [ ] Import Analysis
- [ ] Circular Dependency Detection

---

## Version 1.4

- [ ] Dead Code Detection
- [ ] Unused Functions
- [ ] Unreachable Code

---

## Version 1.5

- [ ] Repository Architecture Visualization
- [ ] Interactive Graphs

---

## Version 2.0

- [ ] FastAPI Backend
- [ ] React Frontend
- [ ] Repository Dashboard
- [ ] Authentication

---

## Version 2.1

- [ ] Git History Intelligence
- [ ] Commit Analysis
- [ ] Blame Analysis
- [ ] PR Summaries

---

## Version 2.2

- [ ] Autonomous Repository Agent
- [ ] Multi-step Planning
- [ ] Automatic Bug Investigation
- [ ] Repository Refactoring Suggestions

---

# 🎯 Why This Project?

Modern repositories contain thousands of files.

Traditional keyword search cannot answer questions like

- "How does authentication work?"
- "Where is this class used?"
- "Explain the request lifecycle."
- "What happens if I change this function?"

This project combines semantic search, AST analysis, and local LLM reasoning to provide repository-level understanding while keeping code private.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Ruthwik Reddy Sama**

GitHub: https://github.com/ruthwik1512

---

⭐ If you found this project interesting, consider giving it a star!
