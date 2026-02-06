# 🤖 RAMESH - AI-Powered Data Collector Agent v2.0

> **R**esearch **A**nd **M**aterial **E**xtraction **S**ystem **H**elper  
> Created by Sajak 🇳🇵

An intelligent agent for building large-scale training datasets from Z-Library (books) and arXiv (research papers).

---

## 🚀 Features

- 📚 **Dual Source Support**: Z-Library books + arXiv papers
- 🤖 **AI-Powered Agent**: GPT-4o powered conversational interface
- ☁️ **Cloud Memory**: Supabase-based duplicate detection (multi-user)
- 🔄 **Auto Account Rotation**: Automatic Z-Library account switching
- 📊 **Bulk Collection**: Download 1000+ papers with curated topics
- 💾 **Resume Support**: Checkpoint/resume for long collections

---

## 📁 Project Structure

```
DataCollectornValidatorAgent/
│
├── main.py                 # 🚀 Main entry point
├── requirements.txt        # Dependencies
│
├── src/                    # Source code
│   ├── core/              # Core modules
│   │   ├── agent.py       # AI agent (GPT-4o)
│   │   ├── memory.py      # Supabase cloud memory
│   │   └── curated_papers.py  # Topic definitions
│   │
│   └── collectors/        # Data collectors
│       ├── arxiv_collector.py  # arXiv paper downloads
│       └── mcp_server.py       # Z-Library book downloads
│
├── config/                 # Configuration
│   ├── .env               # Environment variables (secrets)
│   ├── .env.example       # Template
│   └── accounts.json      # Z-Library accounts
│
├── scripts/               # Utility scripts
│   ├── run_curated_collection.py  # Bulk paper collection
│   ├── setup_supabase.py          # Database setup
│   └── setup_db.py                # Local DB setup
│
├── data/                  # Downloaded data
│   ├── books/            # Z-Library PDFs
│   └── papers/           # arXiv PDFs
│
├── docs/                  # Documentation
│   └── TECHNICAL.md      # Full technical docs
│
└── logs/                  # Runtime logs
    ├── collection_checkpoint.json
    └── collection_report.json
```

---

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Copy template
cp config/.env.example config/.env

# Edit config/.env with your keys:
# - OPENAI_API_KEY
# - SUPABASE_URL
# - SUPABASE_KEY
```

### 3. Run

```bash
# Interactive agent mode
python main.py

# Bulk paper collection (930 curated papers)
python scripts/run_curated_collection.py --dry-run   # Preview
python scripts/run_curated_collection.py             # Start
python scripts/run_curated_collection.py --resume    # Resume
```

---

## 🎯 Usage Examples

### Interactive Mode

```
$ python main.py

Select data source:
1. Z-Library (Books)
2. arXiv (Research Papers)
Enter choice (1/2): 2

You: Download 50 papers on transformer architectures

RAMESH: Namaste! Let me search arXiv for transformer papers...
```

### Bulk Collection

```bash
# See the collection plan
python scripts/run_curated_collection.py --dry-run

# Output:
# 📊 COLLECTION PLAN:
# ├── Foundations: 150 papers
# ├── Architectures: 180 papers
# ├── LLM & NLP: 205 papers
# └── Total: 930 papers

# Start collection
python scripts/run_curated_collection.py

# Interrupted? Resume anytime:
python scripts/run_curated_collection.py --resume
```

---

## 📖 Documentation

See [docs/TECHNICAL.md](docs/TECHNICAL.md) for:
- Full architecture diagrams
- API integrations (arXiv, OpenAI, Supabase)
- Database schema
- Troubleshooting guide

---

## 🛠️ Development

### Add New Collector

1. Create `src/collectors/your_collector.py`
2. Add to `src/collectors/__init__.py`
3. Register tools in `src/core/agent.py`

### Add New Topics

Edit `src/core/curated_papers.py`:
```python
CURATED_TOPICS["your_category"]["Your Topic"] = {
    "queries": ["search query 1", "search query 2"],
    "papers_per_query": 10,
    "total_target": 30,
    "categories": ["cs.LG"]
}
```

---

## 📊 Stats

Check your collection:
```bash
# Count papers
Get-ChildItem data/papers/*.pdf | Measure-Object

# View report
cat logs/collection_report.json
```

---

## 🙏 Credits

- **Creator**: Sajak 🇳🇵
- **Agent Name**: RAMESH (with Nepali humor!)
- **Powered by**: OpenAI GPT-4o, Supabase, arXiv API

---

*"Ramesh never forgets, and now your whole team won't either!"* ☕
