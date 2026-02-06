# RAMESH v2.0 - Technical Documentation 📚🔬

> **Complete Developer Guide for the AI-Powered Data Collector Agent**  
> Created by Sajak 🇳🇵 | Last Updated: February 2026

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [File Structure](#3-file-structure)
4. [Core Components](#4-core-components)
5. [Data Flow](#5-data-flow)
6. [Configuration](#6-configuration)
7. [Database Schema](#7-database-schema)
8. [API Integrations](#8-api-integrations)
9. [Running the System](#9-running-the-system)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Overview

### What is RAMESH?

RAMESH (named with Nepali humor 🇳🇵) is an AI-powered data collection agent designed to build large-scale training datasets for LLMs. It supports two data sources:

| Source | Type | Limit | Authentication |
|--------|------|-------|----------------|
| **Z-Library** | Books (PDFs) | 9 downloads/account | Cookie-based |
| **arXiv** | Research Papers (PDFs) | Unlimited | None required |

### Key Features

- 🤖 **AI Agent Interface**: Conversational UI powered by GPT-4
- ☁️ **Cloud Memory**: Supabase-based shared duplicate detection
- 🔄 **Multi-Account Rotation**: Automatic account switching for Z-Library
- 📊 **Bulk Collection**: Download thousands of papers with curated queries
- 💾 **Checkpoint/Resume**: Interrupt and resume large collections
- 🔒 **SSL Support**: Windows-compatible SSL certificate handling

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAMESH v2.0 ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐      │
│    │   User CLI   │◄───────►│  agent.py    │◄───────►│   OpenAI     │      │
│    │  Interface   │         │ (AI Agent)   │         │   GPT-4o     │      │
│    └──────────────┘         └──────┬───────┘         └──────────────┘      │
│                                    │                                        │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│                    ▼               ▼               ▼                        │
│           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│           │ mcp_server.py│ │arxiv_collector│ │  memory.py   │               │
│           │ (Z-Library)  │ │   (arXiv)    │ │  (Supabase)  │               │
│           └──────┬───────┘ └──────┬───────┘ └──────┬───────┘               │
│                  │                │                │                        │
│                  ▼                ▼                ▼                        │
│           ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│           │  Z-Library   │ │  arXiv API   │ │   Supabase   │               │
│           │   Website    │ │export.arxiv  │ │  PostgreSQL  │               │
│           └──────────────┘ └──────────────┘ └──────────────┘               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              DATA STORAGE

    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │   📁 data/books/     → Z-Library downloaded PDFs                   │
    │   📁 data/papers/    → arXiv downloaded PDFs                       │
    │   📄 resources.json  → Book metadata                               │
    │   📄 arxiv_resources.json → Paper metadata                         │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|---------------|
| `agent.py` | Main AI agent, tool execution, conversation management |
| `mcp_server.py` | Z-Library scraping, book downloads, account rotation |
| `arxiv_collector.py` | arXiv API integration, paper downloads, rate limiting |
| `memory.py` | Supabase client, duplicate detection, embeddings |
| `curated_papers.py` | Predefined topic list for bulk arXiv collection |
| `run_curated_collection.py` | Automated bulk collection runner |

---

## 3. File Structure

```
DataCollectornValidatorAgent/
│
├── 📄 agent.py                    # Main AI agent (GPT-4o powered)
├── 📄 mcp_server.py               # Z-Library downloader
├── 📄 arxiv_collector.py          # arXiv paper collector
├── 📄 memory.py                   # Supabase cloud memory
├── 📄 curated_papers.py           # Curated topic definitions
├── 📄 run_curated_collection.py   # Bulk collection runner
├── 📄 run.py                      # Simple entry point
│
├── 📄 .env                        # Configuration (secrets)
├── 📄 .env.example                # Configuration template
├── 📄 accounts.json               # Z-Library account cookies
├── 📄 requirements.txt            # Python dependencies
│
├── 📄 collection_checkpoint.json  # Resume checkpoint (auto-generated)
├── 📄 collection_report.json      # Collection statistics (auto-generated)
│
├── 📁 data/
│   ├── 📁 books/                  # Downloaded Z-Library PDFs
│   ├── 📁 papers/                 # Downloaded arXiv PDFs
│   ├── 📄 resources.json          # Book metadata
│   └── 📄 arxiv_resources.json    # Paper metadata
│
└── 📄 new_technical_docs.md       # This documentation
```

---

## 4. Core Components

### 4.1 agent.py - The AI Brain

The main conversational agent that orchestrates all operations.

**Key Classes:**
```python
class DataSource(Enum):
    ZLIBRARY = "zlibrary"
    ARXIV = "arxiv"

class LibrarianAgent:
    def __init__(self, source: DataSource = DataSource.ZLIBRARY)
    async def execute_tool(self, tool_name, arguments) -> str
    async def chat(self, user_message: str) -> str
    def switch_source(self, new_source: DataSource)
```

**Flow:**
1. User starts agent → selects data source (Z-Library or arXiv)
2. User sends message → agent processes with GPT-4o
3. GPT-4o may call tools → agent executes tools
4. Results returned → agent responds to user

**Tools Available:**

| Tool | Source | Description |
|------|--------|-------------|
| `download_books` | Z-Library | Search and download books |
| `check_remaining_downloads` | Z-Library | Check account quotas |
| `search_memory` | Both | Search for existing downloads |
| `get_stats` | Both | Get collection statistics |
| `download_papers` | arXiv | Search and download papers |
| `list_arxiv_categories` | arXiv | Show arXiv category codes |
| `search_paper_memory` | arXiv | Search paper database |

---

### 4.2 arxiv_collector.py - Paper Downloader

Handles all arXiv API interactions and PDF downloads.

**Key Classes:**
```python
@dataclass
class ArxivPaper:
    arxiv_id: str          # e.g., "2301.00001"
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]  # e.g., ["cs.LG", "cs.AI"]
    primary_category: str
    published: str
    updated: str
    pdf_url: str
    abs_url: str

class ArxivCollector:
    async def search_papers(query, categories, max_results) -> List[ArxivPaper]
    async def download_pdf(paper: ArxivPaper) -> Optional[str]
    async def collect_papers(query, categories, max_papers) -> dict
```

**Search Query Building:**
```python
# OLD (exact phrase - few results):
search_query = f'all:"{query}"'

# NEW (word-by-word AND - many more results):
words = query.split()
search_query = " AND ".join([f"all:{word}" for word in words])
# Example: "deep learning" → (all:deep AND all:learning)
```

**Rate Limiting:**
- API requests: 3 seconds between calls
- PDF downloads: 1 second between files
- Automatic retry with exponential backoff

**SSL Fix for Windows:**
```python
import certifi
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
```

---

### 4.3 memory.py - Cloud Memory System

Supabase-based shared memory for multi-user duplicate detection.

**Why Supabase?**
- Multiple users need to share the same database
- SQLite is local-only, can't sync across PCs
- Supabase = Free PostgreSQL in the cloud

**Key Functions:**
```python
class AgentMemory:
    # Books (Z-Library)
    def check_duplicate(title, authors) -> dict
    def add_book(title, authors, source, topic, user) -> bool
    def get_stats() -> dict
    def search_similar(query, limit) -> list
    
    # Papers (arXiv)
    def check_paper_duplicate(arxiv_id, title) -> dict
    def add_paper(arxiv_id, title, authors, abstract, categories, user) -> bool
    def get_paper_stats() -> dict
    def search_papers_similar(query, limit) -> list
    
    # Combined
    def get_combined_stats() -> dict
```

**Duplicate Detection:**
1. First check exact match (arxiv_id or normalized_title)
2. If no exact match, compute embedding similarity
3. Similarity > 85% = duplicate

**Embedding Model:** `text-embedding-3-small` (OpenAI)

---

### 4.4 curated_papers.py - Topic Definitions

Predefined topics for quality-focused bulk collection.

**Structure:**
```python
CURATED_TOPICS = {
    "foundations": {
        "Neural Network Fundamentals": {
            "queries": [
                "backpropagation neural network learning",
                "universal approximation theorem neural",
                "deep learning representation learning"
            ],
            "papers_per_query": 10,
            "total_target": 30,
            "categories": ["cs.LG", "cs.NE"],
            "year_range": (2010, 2024)
        },
        # ... more topics
    },
    "architectures": { ... },
    "llm_nlp": { ... },
    "vision_multimodal": { ... },
    "training_efficiency": { ... },
    "evaluation_safety": { ... }
}

SURVEY_QUERIES = [
    {"query": "survey deep learning neural network", "max_papers": 20, "categories": ["cs.LG"]},
    # ... 9 more survey queries
]
```

**Categories (23 total):**
- Foundations: 5 topics (150 papers)
- Architectures: 6 topics (180 papers)
- LLM & NLP: 7 topics (205 papers)
- Vision & Multimodal: 4 topics (100 papers)
- Training Efficiency: 4 topics (100 papers)
- Evaluation & Safety: 2 topics (50 papers)
- Surveys: 10 queries (145 papers)

**Total Target: ~930 high-quality papers**

---

### 4.5 run_curated_collection.py - Bulk Runner

Automated collection with checkpoint/resume capability.

**Key Class:**
```python
class CuratedCollectionRunner:
    def __init__(self, user_name: str = "sajak")
    async def collect_topic(category, topic_name, config) -> int
    async def collect_surveys() -> int
    async def run_full_collection()
    
    # Checkpoint management
    def _load_checkpoint() -> dict
    def _save_checkpoint()
    def _save_report()
```

**Checkpoint Format:**
```json
{
  "completed_topics": [
    "foundations:Neural Network Fundamentals",
    "foundations:Optimization in Deep Learning"
  ],
  "completed_surveys": [
    "survey deep learning neural network"
  ]
}
```

**Usage:**
```bash
# Show plan without downloading
python run_curated_collection.py --dry-run

# Start fresh collection
python run_curated_collection.py

# Resume from checkpoint
python run_curated_collection.py --resume

# Specific user name
python run_curated_collection.py --user yourname
```

---

## 5. Data Flow

### 5.1 Z-Library Book Download Flow

```
User Request
    │
    ▼
┌─────────────────┐
│   agent.py      │
│ parse_intent()  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  memory.py      │────►│   Supabase      │
│ check_duplicate │     │ (cloud lookup)  │
└────────┬────────┘     └─────────────────┘
         │
         │ (if not duplicate)
         ▼
┌─────────────────┐
│ mcp_server.py   │
│ search_zlibrary │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Z-Library     │
│   Website       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ download_book() │
│ → data/books/   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  memory.py      │────►│   Supabase      │
│ add_book()      │     │ (save record)   │
└─────────────────┘     └─────────────────┘
```

### 5.2 arXiv Paper Collection Flow

```
User/Script Request
    │
    ▼
┌─────────────────────┐
│ arxiv_collector.py  │
│ collect_papers()    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────┐
│    memory.py        │────►│    Supabase     │
│ check_paper_dup()   │     │ (cloud lookup)  │
└─────────┬───────────┘     └─────────────────┘
          │
          │ (if not duplicate)
          ▼
┌─────────────────────┐
│   arXiv API         │
│ export.arxiv.org    │
│ /api/query          │
└─────────┬───────────┘
          │
          │ (XML response)
          ▼
┌─────────────────────┐
│ _parse_arxiv_resp() │
│ → ArxivPaper list   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ download_pdf()      │
│ → data/papers/      │
│ (rate limited)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌─────────────────┐
│    memory.py        │────►│    Supabase     │
│ add_paper()         │     │ (save record)   │
└─────────────────────┘     └─────────────────┘
```

### 5.3 Bulk Collection Flow (run_curated_collection.py)

```
Start Script
    │
    ▼
┌──────────────────────┐
│ Load checkpoint      │
│ (or start fresh)     │
└──────────┬───────────┘
           │
           ▼
    ┌──────────────┐
    │ For each     │◄─────────────────────┐
    │ category     │                      │
    └──────┬───────┘                      │
           │                              │
           ▼                              │
    ┌──────────────┐                      │
    │ For each     │◄────────────┐        │
    │ topic        │             │        │
    └──────┬───────┘             │        │
           │                     │        │
           │ (skip if in         │        │
           │  checkpoint)        │        │
           ▼                     │        │
    ┌──────────────┐             │        │
    │ For each     │◄───┐        │        │
    │ query        │    │        │        │
    └──────┬───────┘    │        │        │
           │            │        │        │
           ▼            │        │        │
    ┌──────────────┐    │        │        │
    │ ArxivCollect │    │        │        │
    │ .collect()   │    │        │        │
    └──────┬───────┘    │        │        │
           │            │        │        │
           └────────────┘        │        │
                                 │        │
    ┌──────────────┐             │        │
    │ Save topic   │─────────────┘        │
    │ to checkpoint│                      │
    └──────────────┘                      │
                                          │
    ┌──────────────┐                      │
    │ Next category│──────────────────────┘
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Collect      │
    │ surveys      │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ Save report  │
    │ Print stats  │
    └──────────────┘
```

---

## 6. Configuration

### 6.1 Environment Variables (.env)

```bash
# === REQUIRED ===
OPENAI_API_KEY=sk-...              # For GPT-4o and embeddings

# === SUPABASE (Cloud Memory) ===
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIs...  # anon/public key

# === Z-LIBRARY ===
BOOK_DATA_FOLDER=data/books
RESOURCES_FILE=data/resources.json
MAX_DOWNLOADS_PER_ACCOUNT=9

# === ARXIV ===
ARXIV_DATA_FOLDER=data/papers
ARXIV_RESOURCES_FILE=data/arxiv_resources.json
ARXIV_API_RATE_LIMIT=3.0           # seconds between API calls
ARXIV_PDF_DELAY=1.0                # seconds between PDF downloads
ARXIV_MAX_PER_QUERY=100            # max results per API query
ARXIV_MAX_RETRIES=3
ARXIV_DOWNLOAD_TIMEOUT=120         # seconds

# === AGENT ===
MAX_CONVERSATION_HISTORY=20
```

### 6.2 Z-Library Accounts (accounts.json)

```json
{
    "accounts": [
        {
            "email": "user1@email.com",
            "cookies": {
                "remix_userid": "12345678",
                "remix_userkey": "abcdef123..."
            }
        },
        {
            "email": "user2@email.com",
            "cookies": {
                "remix_userid": "87654321",
                "remix_userkey": "fedcba321..."
            }
        }
    ]
}
```

**How to get cookies:**
1. Log into Z-Library in browser
2. Open DevTools → Application → Cookies
3. Copy `remix_userid` and `remix_userkey` values

---

## 7. Database Schema

### 7.1 Supabase Tables

**book_memory** (Z-Library books):
```sql
CREATE TABLE book_memory (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,      -- lowercase, no punctuation
    authors TEXT,
    source TEXT,                          -- 'zlibrary'
    search_topic TEXT,                    -- original search query
    embedding FLOAT8[],                   -- 1536-dim vector
    downloaded_by TEXT,                   -- user who downloaded
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_normalized_title ON book_memory(normalized_title);
CREATE INDEX idx_downloaded_by ON book_memory(downloaded_by);
```

**paper_memory** (arXiv papers):
```sql
CREATE TABLE paper_memory (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,       -- e.g., "2301.00001"
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    categories TEXT,                      -- e.g., "cs.LG, cs.AI"
    embedding FLOAT8[],                   -- 1536-dim vector
    downloaded_by TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_paper_arxiv_id ON paper_memory(arxiv_id);
CREATE INDEX idx_paper_normalized_title ON paper_memory(normalized_title);
CREATE INDEX idx_paper_downloaded_by ON paper_memory(downloaded_by);
```

### 7.2 Local Files

**resources.json** (Book metadata):
```json
{
    "books": [
        {
            "title": "Deep Learning",
            "authors": "Ian Goodfellow",
            "filename": "deep_learning_goodfellow.pdf",
            "topic": "deep learning fundamentals",
            "downloaded_at": "2026-02-04T10:30:00"
        }
    ]
}
```

**arxiv_resources.json** (Paper metadata):
```json
{
    "papers": [
        {
            "arxiv_id": "2301.00001",
            "title": "Attention Is All You Need",
            "authors": ["Vaswani", "Shazeer", "..."],
            "categories": ["cs.CL", "cs.LG"],
            "filename": "2301_00001_Attention_Is_All_You_Need.pdf",
            "downloaded_at": "2026-02-04T10:30:00"
        }
    ]
}
```

---

## 8. API Integrations

### 8.1 arXiv API

**Base URL:** `http://export.arxiv.org/api/query`

**Query Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `search_query` | Search terms | `all:deep AND all:learning` |
| `start` | Pagination offset | `0` |
| `max_results` | Results per page | `100` (max 2000) |
| `sortBy` | Sort field | `relevance`, `submittedDate` |
| `sortOrder` | Sort direction | `ascending`, `descending` |

**Example Request:**
```
http://export.arxiv.org/api/query?search_query=(all:transformer AND all:attention) AND (cat:cs.LG OR cat:cs.CL)&start=0&max_results=10&sortBy=relevance
```

**Response:** Atom XML feed

**Rate Limits:**
- 1 request per 3 seconds (enforced in code)
- Be polite or get IP banned!

### 8.2 OpenAI API

**Used for:**
1. GPT-4o: Agent conversations and tool calls
2. text-embedding-3-small: Duplicate detection embeddings

**Models:**
| Purpose | Model | Cost |
|---------|-------|------|
| Agent Chat | gpt-4o | ~$5/1M tokens |
| Embeddings | text-embedding-3-small | ~$0.02/1M tokens |

### 8.3 Supabase REST API

**Authentication:**
```python
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}
```

**PostgREST Queries:**
```python
# Insert
client.from_("paper_memory").insert({...}).execute()

# Select with filter
client.from_("paper_memory").select("*").eq("arxiv_id", "2301.00001").execute()

# Select all
client.from_("book_memory").select("*").execute()
```

---

## 9. Running the System

### 9.1 Initial Setup

```bash
# 1. Create virtual environment
python -m venv LLMenv
.\LLMenv\Scripts\Activate.ps1  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment template
copy .env.example .env

# 4. Fill in .env with your keys

# 5. Create Supabase tables (run SQL from setup_supabase.py)
python setup_supabase.py
```

### 9.2 Interactive Mode (Agent Chat)

```bash
# Start the agent
python run.py

# Or directly
python agent.py
```

**Example session:**
```
Select data source:
1. Z-Library (Books)
2. arXiv (Research Papers)
Enter choice (1/2): 2

You: Download 100 papers on transformers and attention mechanisms

RAMESH: Namaste! Let me collect those papers for you...
```

### 9.3 Bulk Collection Mode

```bash
# See the collection plan
python run_curated_collection.py --dry-run

# Start collection
python run_curated_collection.py

# Resume after interruption
python run_curated_collection.py --resume
```

### 9.4 Common Commands

```bash
# Check downloaded paper count
Get-ChildItem data/papers/*.pdf | Measure-Object

# Check downloaded book count
Get-ChildItem data/books/*.pdf | Measure-Object

# View collection report
cat collection_report.json

# Clear checkpoint (start fresh)
Remove-Item collection_checkpoint.json
```

---

## 10. Troubleshooting

### Issue: SSL Certificate Error on Windows

**Error:**
```
SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate
```

**Solution:**
```bash
pip install certifi
```

The code automatically uses certifi for SSL context.

### Issue: arXiv Returns 0 Papers

**Cause:** Exact phrase matching is too strict.

**Solution:** The code now uses word-by-word AND matching:
```python
# Instead of: all:"deep learning transformers"
# Now uses:   (all:deep AND all:learning AND all:transformers)
```

### Issue: Supabase Connection Failed

**Check:**
1. `.env` has correct `SUPABASE_URL` and `SUPABASE_KEY`
2. Tables exist (run `setup_supabase.py`)
3. RLS policies allow inserts

### Issue: Z-Library Download Limit Reached

**Solution:** Add more accounts to `accounts.json`. The system auto-rotates.

### Issue: Collection Stopped Mid-Way

**Solution:** Use `--resume` flag:
```bash
python run_curated_collection.py --resume
```

Progress is saved in `collection_checkpoint.json`.

### Issue: Too Many Duplicate Papers

**Check:**
- Supabase `paper_memory` table has embeddings
- `SIMILARITY_THRESHOLD` in memory.py (default: 0.85)

---

## Quick Reference

### Key Commands

| Action | Command |
|--------|---------|
| Start agent | `python run.py` |
| Bulk collect | `python run_curated_collection.py` |
| Resume collection | `python run_curated_collection.py --resume` |
| Dry run | `python run_curated_collection.py --dry-run` |
| Check papers | `Get-ChildItem data/papers/*.pdf \| Measure-Object` |
| View report | `cat collection_report.json` |

### File Locations

| Data | Location |
|------|----------|
| Books PDFs | `data/books/` |
| Paper PDFs | `data/papers/` |
| Book metadata | `data/resources.json` |
| Paper metadata | `data/arxiv_resources.json` |
| Checkpoint | `collection_checkpoint.json` |
| Report | `collection_report.json` |

### Important Limits

| Limit | Value |
|-------|-------|
| arXiv API rate | 3 seconds between requests |
| arXiv max per query | 2000 results |
| Z-Library per account | 9 downloads |
| Embedding similarity threshold | 85% |

---

**Created by Sajak 🇳🇵**  
*"Ramesh never forgets, and now your whole team won't either!"*
