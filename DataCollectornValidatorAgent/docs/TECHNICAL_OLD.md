# 🛠️ RAMESH Technical Documentation

> **Version**: 1.1  
> **Last Updated**: January 2026  
> **Author**: Sajak  
> **For**: Developers & Technical Team Members

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Module Reference](#3-module-reference)
4. [Data Flow & Execution](#4-data-flow--execution)
5. [Database Schema](#5-database-schema)
6. [API Reference](#6-api-reference)
7. [Configuration Guide](#7-configuration-guide)
8. [Dependencies](#8-dependencies)
9. [Deployment Guide](#9-deployment-guide)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Overview

### 1.1 What is RAMESH?

RAMESH (**R**etrieve **A**nd **M**anage **E**-books with **S**emantic **H**ashing) is an AI-powered data collection agent that:

- **Downloads books** from Z-Library using Playwright browser automation
- **Avoids duplicates** using semantic embedding similarity (85% threshold)
- **Shares memory** across 4 team members via Supabase cloud database
- **Makes decisions** using GPT-4o for intelligent topic parsing and search strategy

### 1.2 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│   Sajak's PC        Siddarth's PC        Ronish's PC        Dipsan's PC      │
│        │                  │                   │                  │           │
│        └──────────────────┴─────────┬─────────┴──────────────────┘           │
│                                     │                                         │
│                              ┌──────▼──────┐                                  │
│                              │  agent.py   │  ← GPT-4o Decision Making        │
│                              │  (Entry)    │                                  │
│                              └──────┬──────┘                                  │
├─────────────────────────────────────┼────────────────────────────────────────┤
│                              CORE LAYER                                       │
├─────────────────────────────────────┼────────────────────────────────────────┤
│        ┌────────────────────────────┼─────────────────────────────┐          │
│        │                            │                             │          │
│        ▼                            ▼                             ▼          │
│  ┌──────────┐               ┌──────────────┐              ┌─────────────┐    │
│  │ memory.py│               │mcp_server.py │              │accounts.json│    │
│  │ (Cloud)  │               │ (Playwright) │              │ (Creds)     │    │
│  └────┬─────┘               └──────┬───────┘              └─────────────┘    │
│       │                            │                                          │
├───────┼────────────────────────────┼─────────────────────────────────────────┤
│       │              EXTERNAL SERVICES                                        │
├───────┼────────────────────────────┼─────────────────────────────────────────┤
│       ▼                            ▼                                          │
│  ┌──────────┐               ┌──────────────┐              ┌─────────────┐    │
│  │ SUPABASE │               │  Z-LIBRARY   │              │   OPENAI    │    │
│  │PostgreSQL│               │    Web       │              │   API       │    │
│  └──────────┘               └──────────────┘              └─────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| AI Brain | GPT-4o | Decision making, tool calling |
| Embeddings | text-embedding-3-small | Semantic duplicate detection |
| Metadata Extraction | GPT-4o-mini | Fast title/author parsing |
| Browser Automation | Playwright (Async) | Z-Library navigation |
| Cloud Database | Supabase PostgreSQL | Shared team memory |
| HTTP Client | PostgREST | Supabase REST API access |

---

## 2. Architecture Deep Dive

### 2.1 Module Relationship Diagram

```
                    ┌─────────────────────────────────────────┐
                    │              agent.py                   │
                    │  ┌──────────────────────────────────┐   │
                    │  │      LibrarianAgent Class        │   │
                    │  │  - GPT-4o conversation           │   │
                    │  │  - Tool definitions              │   │
                    │  │  - Account rotation              │   │
                    │  └────────────┬─────────────────────┘   │
                    └───────────────┼─────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐  ┌─────────────────┐   ┌─────────────────┐
    │    memory.py    │  │  mcp_server.py  │   │  accounts.json  │
    │  ┌───────────┐  │  │  ┌───────────┐  │   │                 │
    │  │AgentMemory│  │  │  │core_down- │  │   │ [{credentials}] │
    │  │   Class   │  │  │  │load_logic │  │   │                 │
    │  └─────┬─────┘  │  │  └─────┬─────┘  │   └─────────────────┘
    │        │        │  │        │        │
    │        ▼        │  │        ▼        │
    │  ┌───────────┐  │  │  ┌───────────┐  │
    │  │Supabase   │  │  │  │Playwright │  │
    │  │PostgREST  │  │  │  │Browser    │  │
    │  └───────────┘  │  │  └─────┬─────┘  │
    └─────────────────┘  │        │        │
              │          │        ▼        │
              │          │ ┌─────────────┐ │
              │          │ │check_site_  │ │
              │          │ │status.py    │ │
              │          │ └─────┬───────┘ │
              │          └───────┼─────────┘
              ▼                  ▼
    ┌─────────────────┐  ┌─────────────────┐
    │   SUPABASE      │  │   Z-LIBRARY     │
    │   Cloud DB      │  │   Website       │
    └─────────────────┘  └─────────────────┘
```

### 2.2 Data Flow Sequence

```
User Request: "Get me books on machine learning"
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 1. agent.py receives user input                                  │
│    └── Adds to conversation: [{"role": "user", "content": ...}] │
└──────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. GPT-4o processes with tool definitions                        │
│    └── Returns tool_call: download_books(topic="...", max=3)     │
└──────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. agent.py.execute_tool() calls mcp_server.core_download_logic()│
│    └── Passes: topic, account, max_books, memory, user_name     │
└──────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. core_download_logic() checks site availability first          │
│    └── Calls check_site_status() (HEAD request with fallback)    │
│    └── If NOT OK: Returns error immediately (no browser launch)  │
│    └── If OK: Proceeds to launch Playwright browser              │
└──────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. Playwright browser automation                                 │
│    └── Sets cookies for Z-Library auth                           │
│    └── Navigates to search: z-lib.sk/s/{topic}                   │
└──────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. For each <z-bookcard> element found:                          │
│    ├── Extract: title, author, download_path, extension          │
│    ├── Call memory.check_duplicate(title, author)                │
│    │   └── Generates embedding via OpenAI                        │
│    │   └── Compares cosine similarity vs all stored books        │
│    │   └── Returns: {is_duplicate: bool, similarity: float}      │
│    ├── If NOT duplicate:                                         │
│    │   └── Download file via page.expect_download()              │
│    │   └── Clean metadata via GPT-4o-mini                        │
│    │   └── Save to local folder                                  │
│    │   └── Call memory.add_book() → Supabase                     │
│    └── If IS duplicate: Skip with log message                    │
└──────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────┐
│ 7. Return results to agent → GPT-4o → User response              │
│    └── If site was down: Agent explains issue clearly            │
│    └── If 0 books found: Agent suggests different search terms   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Reference

### 3.1 `agent.py` - Main Entry Point (455 lines)

The brain of RAMESH. Handles user interaction, GPT-4o conversation, and tool orchestration.

#### Key Constants

```python
RAMESH_BANNER       # ASCII art banner displayed on startup
SYSTEM_PROMPT       # GPT-4o personality and instructions
TOOLS               # OpenAI function calling definitions
```

#### Class: `LibrarianAgent`

```python
class LibrarianAgent:
    """
    The main AI agent that orchestrates book downloads.
    
    Attributes:
        user_name (str): Current user for tracking downloads
        accounts (list): Z-Library account credentials
        current_account_idx (int): Which account is being used
        downloads_on_current_account (int): Counter for rate limiting
        max_per_account (int): 9 books per account limit
        session_downloads (list): Tracks downloads in current session
        conversation (list): GPT-4o message history
        memory (AgentMemory): Supabase cloud memory instance
    """
```

##### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `__init__(user_name)` | Initialize agent with user identification | None |
| `_load_accounts()` | Load credentials from `accounts.json` | list[dict] |
| `current_account` | Property: Get current account dict | dict or None |
| `remaining_downloads` | Property: Calculate remaining downloads | int |
| `_rotate_account_if_needed()` | Switch to next account when limit hit | bool |
| `execute_tool(name, args)` | Execute a tool and return result | str |
| `chat(user_message)` | Send message to GPT-4o, handle tool calls | str |
| `reset_session()` | Clear session state (keeps history) | None |

##### Tool Definitions

```python
TOOLS = [
    {
        "name": "download_books",
        "parameters": {
            "topic": str,       # Search query
            "max_books": int    # 1-9, default 3
        }
    },
    {"name": "check_remaining_downloads"},  # No params
    {"name": "list_downloaded_books"},       # No params
    {
        "name": "search_memory",
        "parameters": {"query": str}         # Search similar books
    },
    {"name": "get_memory_stats"}             # No params
]
```

##### User Greeting Logic (Easter Egg 🥚)

```python
# Located in main()
if name_lower == "sajak":
    greeting_suffix = "dai"       # Respectful
elif name_lower == "dipsan":
    greeting_suffix = "didi"      # Prank: "sister" instead of "bro"
elif name_lower == "siddarth":
    greeting_suffix = "muji"      # Savage Nepali slang
elif name_lower == "ronish":
    greeting_suffix = "Please"    # Extra polite prank
```

---

### 3.2 `memory.py` - Cloud Memory System (397 lines)

Handles Supabase PostgreSQL connection and semantic duplicate detection.

#### Key Constants

```python
SIMILARITY_THRESHOLD = 0.85  # 85% = duplicate
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
```

#### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `get_postgrest()` | Get/create Supabase REST client | SyncPostgrestClient |
| `init_memory_db()` | Print SQL to create table | None |
| `get_embedding(text)` | Generate 1536-dim embedding | list[float] |
| `cosine_similarity(a, b)` | Calculate vector similarity | float (0-1) |

##### `get_embedding()` Implementation

```python
def get_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dimensions
        input=text
    )
    return response.data[0].embedding
```

##### `cosine_similarity()` Implementation

```python
def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

#### Class: `AgentMemory`

```python
class AgentMemory:
    """
    Cloud-based Memory System for RAMESH Agent.
    Uses Supabase for multi-user shared memory.
    
    Attributes:
        client (SyncPostgrestClient): Supabase REST client
        _cache (dict): Local cache for speed (unused currently)
    """
```

##### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `check_duplicate(title, authors)` | Check if book exists in memory | dict |
| `_check_exact_match(title)` | Fallback: exact title match | dict |
| `add_book(title, authors, source, topic, user)` | Add book after download | bool |
| `get_all_books()` | Fetch all books from memory | list[dict] |
| `get_books_by_topic(topic)` | Filter books by search topic | list[dict] |
| `get_stats()` | Get memory statistics | dict |
| `search_similar(query, limit)` | Semantic search for books | list[dict] |

##### `check_duplicate()` Return Schema

```python
{
    "is_duplicate": bool,          # True if similarity >= 0.85
    "similar_book": {              # Only if is_duplicate=True
        "id": int,
        "title": str,
        "authors": str,
        "downloaded_by": str,
        "similarity": float
    } or None,
    "similarity": float            # Max similarity found (0-1)
}
```

##### `get_stats()` Return Schema

```python
{
    "total_books": int,
    "by_user": {
        "sajak": 10,
        "siddarth": 5,
        ...
    },
    "top_topics": {
        "machine learning": 8,
        "deep learning": 5,
        ...
    }
}
```

---

### 3.3 `check_site_status.py` - Site Availability Check (35 lines)

Lightweight pre-flight check to verify Z-Library is accessible before launching Playwright.

#### Key Constants

```python
URL = "https://z-lib.sk/"  # Must match BASE_URL in mcp_server.py
TIMEOUT = 10               # Request timeout in seconds
```

#### Function: `check_site_status()`

```python
def check_site_status() -> str:
    """
    Check if Z-Library site is accessible.
    Uses HEAD first (lightweight), falls back to GET if HEAD fails.
    
    Returns:
        "OK" - Site is accessible
        "NOT OK: <reason>" - Site is down with specific error
    """
```

##### Return Values

| Return | Meaning |
|--------|---------|  
| `"OK"` | Site is up and responding |
| `"NOT OK: Request timed out"` | Site didn't respond in 10s |
| `"NOT OK: Connection failed - ..."` | Network/DNS error |
| `"NOT OK: Site is temporarily unavailable (503)"` | Site is down for maintenance |
| `"NOT OK: Access forbidden (403)"` | Cloudflare/WAF blocking |
| `"NOT OK: HTTP response code {code}"` | Other HTTP errors |

##### Implementation Logic

```python
# 1. Try HEAD request first (faster, less bandwidth)
response = requests.head(URL, timeout=TIMEOUT, allow_redirects=True)

# 2. Some sites block HEAD, fallback to GET on 4xx/5xx
if response.status_code >= 400:
    response = requests.get(URL, timeout=TIMEOUT, allow_redirects=True)

# 3. Return status based on response code
if response.status_code == 200:
    return "OK"
```

---

### 3.4 `mcp_server.py` - Download Engine (328 lines)

Core Playwright automation for Z-Library book downloads.

#### Key Constants

```python
BASE_URL = "https://z-lib.sk"
DOWNLOAD_FOLDER = os.path.abspath(os.getenv("DATA_FOLDER", "data/books"))
RESOURCES_FILE = os.getenv("RESOURCES_FILE", "data/resources.json")
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
DOWNLOAD_COOLDOWN = int(os.getenv("DOWNLOAD_COOLDOWN", "40"))  # seconds
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # seconds
```

#### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `clean_metadata_with_llm(raw_text)` | Extract title/author via GPT-4o-mini | dict |
| `save_to_json_file(metadata)` | Append to local JSON database | None |
| `log_download_to_db(meta, filename, account)` | Log with timestamp | None |
| `is_duplicate(title, db)` | Legacy local duplicate check | bool |
| `core_download_logic(...)` | Main download orchestration | tuple |

##### `clean_metadata_with_llm()` Implementation

```python
def clean_metadata_with_llm(raw_text: str) -> dict:
    """
    Uses GPT-4o-mini to extract clean metadata from messy book titles.
    
    Input:  "Introduction to Machine Learning (3rd ed.) by Ethem Alpaydin, PhD"
    Output: {
        "title": "Introduction to Machine Learning",
        "author": "Ethem Alpaydin",
        "edition": "3rd",
        "category": "Machine Learning"
    }
    """
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": METADATA_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text}
        ]
    )
    return json.loads(response.choices[0].message.content)
```

##### `core_download_logic()` Signature

```python
async def core_download_logic(
    topic: str,                    # Search query
    account: dict = None,          # Z-Library credentials
    max_books: int = 9,            # Download limit
    memory: AgentMemory = None,    # Cloud memory instance
    user_name: str = "unknown"     # For tracking
) -> tuple:
    """
    Main download orchestration function.
    
    IMPORTANT: Calls check_site_status() FIRST before launching browser.
    If site is down, returns immediately without wasting resources.
    
    Returns:
        (message: str, count: int, downloaded_books: list[dict])
        
    Example returns:
        ("Site status check failed: ...", 0, [])  # Site down
        ("Download & Indexing Complete.", 5, [...])  # Success
    """
```

##### Pre-flight Site Check

```python
# First check if site is accessible (before launching browser)
status = check_site_status()
if not status.startswith("OK"):
    return (f"Site status check failed: {status}", 0, [])

# Only launch Playwright if site is up
async with async_playwright() as p:
    # ... browser automation
```

##### Z-Library DOM Structure

```html
<!-- Z-Library uses custom web components -->
<z-bookcard 
    class="ready"
    id="12345"
    href="/book/12345/title-slug"
    download="/dl/12345/hash"
    extension="pdf"
>
    <div slot="title">Book Title Here</div>
    <div slot="author">Author Name</div>
</z-bookcard>
```

##### Download Flow within `core_download_logic()`

```python
# 1. Launch browser with Playwright
async with async_playwright() as p:
    browser = await p.chromium.launch(headless=False)
    context = await browser.new_context()
    
    # 2. Set authentication cookies
    await context.add_cookies([
        {"name": "remix_userid", "value": account["remix_userid"], ...},
        {"name": "remix_userkey", "value": account["remix_userkey"], ...}
    ])
    
    # 3. Navigate to search
    page = await context.new_page()
    await page.goto(f"{BASE_URL}/s/{topic}")
    
    # 4. Wait for custom elements
    await page.wait_for_selector("z-bookcard.ready", timeout=15000)
    
    # 5. Extract book data
    bookcards = await page.locator("z-bookcard.ready").all()
    
    for bookcard in bookcards:
        # Extract attributes
        download_path = await bookcard.get_attribute("download")
        title = await bookcard.locator("div[slot='title']").inner_text()
        
        # Check memory for duplicates
        if memory.check_duplicate(title, author)["is_duplicate"]:
            continue  # Skip
        
        # Download via JavaScript navigation (avoids "Download starting" page)
        async with page.expect_download() as download_info:
            await page.evaluate(f"window.location.href = '{url}'")
        
        # Save file
        download = await download_info.value
        await download.save_as(save_path)
        
        # Add to shared memory
        memory.add_book(title, author, "Z-Library", topic, user_name)
```

---

### 3.5 `accounts.json` - Credentials Store

```json
[
    {
        "name": "Account_1",           // Display name
        "remix_userid": "46789364",    // Z-Library user ID (cookie)
        "remix_userkey": "b257583e...", // Z-Library auth key (cookie)
        "daily_limit": 9               // Books per day limit
    },
    {
        "name": "Account_2",
        // ... more accounts for rotation
    }
]
```

**Security Note**: This file contains sensitive credentials. Do NOT commit to version control.

---

## 4. Data Flow & Execution

### 4.1 Complete Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  USER INPUT: "Build me a dataset for NLP and transformers"                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  GPT-4o REASONING:                                                          │
│  "User wants NLP books. I should break this into specific topics:           │
│   1. 'NLP natural language processing tutorial' - 3 books                   │
│   2. 'Transformer deep learning attention mechanism' - 3 books              │
│   3. 'BERT GPT language models' - 3 books                                   │
│  I'll call download_books for each topic."                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌────────────┐  ┌────────────┐  ┌────────────┐
            │ Tool Call 1│  │ Tool Call 2│  │ Tool Call 3│
            │topic="NLP" │  │topic="Trans│  │topic="BERT"│
            │max_books=3 │  │former"     │  │max_books=3 │
            └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
                  │               │               │
                  └───────────────┼───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   execute_tool() Loop       │
                    │   (Sequential Execution)    │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │     core_download_logic()   │
                    │  ┌──────────────────────┐   │
                    │  │ For each z-bookcard: │   │
                    │  │ 1. Extract metadata  │   │
                    │  │ 2. Check memory ─────│───┼──► Supabase Query
                    │  │ 3. Download if new   │   │
                    │  │ 4. Save to memory ───│───┼──► Supabase Insert
                    │  │ 5. Wait 40 seconds   │   │
                    │  └──────────────────────┘   │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  GPT-4o FINAL RESPONSE:     │
                    │  "Ramesh has completed!     │
                    │   Downloaded 9 books:       │
                    │   - 3 on NLP fundamentals   │
                    │   - 3 on Transformers       │
                    │   - 3 on BERT/GPT models"   │
                    └─────────────────────────────┘
```

### 4.2 Duplicate Detection Flow

```
New Book: "Introduction to Machine Learning" by Ethem Alpaydin
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Generate Embedding                                                 │
│  ─────────────────────────                                                  │
│  Input: "Introduction to Machine Learning by Ethem Alpaydin"                │
│  Model: text-embedding-3-small                                              │
│  Output: [0.023, -0.156, 0.089, ...] (1536 dimensions)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Fetch All Stored Embeddings from Supabase                          │
│  ─────────────────────────────────────────────────                          │
│  Query: SELECT id, title, authors, embedding, downloaded_by FROM book_memory│
│  Result: [{embedding: [...], title: "ML Book 1", ...}, ...]                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Calculate Cosine Similarity for Each                               │
│  ─────────────────────────────────────────────                              │
│                                                                             │
│  cosine_similarity(new_embedding, stored_embedding[0]) = 0.72               │
│  cosine_similarity(new_embedding, stored_embedding[1]) = 0.91  ← HIGHEST    │
│  cosine_similarity(new_embedding, stored_embedding[2]) = 0.45               │
│  ...                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Compare to Threshold                                               │
│  ────────────────────────────                                               │
│                                                                             │
│  max_similarity = 0.91                                                      │
│  SIMILARITY_THRESHOLD = 0.85                                                │
│                                                                             │
│  0.91 >= 0.85 → IS DUPLICATE ✓                                              │
│                                                                             │
│  Return: {                                                                  │
│      "is_duplicate": True,                                                  │
│      "similar_book": {"title": "Machine Learning: A Probabilistic...", ...}│
│      "similarity": 0.91                                                     │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Database Schema

### 5.1 Supabase Table: `book_memory`

```sql
CREATE TABLE book_memory (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,        -- Lowercase for exact matching
    authors TEXT,
    source TEXT,                           -- "Z-Library"
    search_topic TEXT,                     -- Original search query
    embedding FLOAT8[],                    -- 1536-dimension vector
    downloaded_by TEXT,                    -- User who downloaded
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_normalized_title ON book_memory(normalized_title);
CREATE INDEX idx_downloaded_by ON book_memory(downloaded_by);
```

### 5.2 Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Auto-incrementing primary key |
| `title` | TEXT | Original book title |
| `normalized_title` | TEXT | Lowercase title for fallback matching |
| `authors` | TEXT | Author name(s) |
| `source` | TEXT | Always "Z-Library" currently |
| `search_topic` | TEXT | What the user searched for |
| `embedding` | FLOAT8[] | 1536-dimension OpenAI embedding |
| `downloaded_by` | TEXT | Username of downloader |
| `created_at` | TIMESTAMPTZ | Auto-set on insert |

### 5.3 Local JSON Schema: `data/resources.json`

```json
{
    "resources": [
        {
            "title": "Clean Book Title",
            "author": "Author Name",
            "edition": "3rd",
            "category": "Machine Learning",
            "filename": "Clean_Book_Title.pdf",
            "account": "Account_1",
            "timestamp": "2025-01-15T10:30:00Z"
        }
    ]
}
```

---

## 6. API Reference

### 6.1 OpenAI API Calls

#### GPT-4o (Agent Reasoning)
```python
# Location: agent.py
client.chat.completions.create(
    model="gpt-4o",
    messages=conversation,
    tools=TOOLS,
    tool_choice="auto"
)
```
- **Purpose**: Main agent reasoning and tool calling
- **Cost**: ~$10/1M tokens (input), ~$30/1M (output)

#### GPT-4o-mini (Metadata Cleaning)
```python
# Location: mcp_server.py
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": METADATA_SYSTEM_PROMPT},
        {"role": "user", "content": raw_text}
    ]
)
```
- **Purpose**: Fast metadata extraction from titles
- **Cost**: ~$0.15/1M tokens (much cheaper)

#### text-embedding-3-small
```python
# Location: memory.py
client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)
```
- **Purpose**: Generate semantic embeddings
- **Dimensions**: 1536
- **Cost**: ~$0.02/1M tokens

### 6.2 Supabase API Calls

All calls use PostgREST client:

```python
# SELECT
client.from_("book_memory").select("*").execute()

# INSERT
client.from_("book_memory").insert(data).execute()

# FILTER
client.from_("book_memory").select("*").eq("column", value).execute()
client.from_("book_memory").select("*").ilike("column", f"%{pattern}%").execute()

# ORDER
client.from_("book_memory").select("*").order("created_at", desc=True).execute()
```

---

## 7. Configuration Guide

### 7.1 Environment Variables (`.env`)

```bash
# OpenAI API Key (required)
OPENAI_API_KEY=sk-proj-...

# Supabase Cloud Database (required)
SUPABASE_URL=https://wtyioljwaeetlelkguti.supabase.co
SUPABASE_KEY=KEY
```

### 7.2 `accounts.json` Configuration

```json
[
    {
        "name": "Account_1",
        "remix_userid": "YOUR_USERID",
        "remix_userkey": "YOUR_USERKEY",
        "daily_limit": 9
    }
]
```

**How to get Z-Library cookies:**
1. Login to Z-Library in browser
2. Open DevTools → Application → Cookies
3. Copy `remix_userid` and `remix_userkey` values

### 7.3 Adjustable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | memory.py | 0.85 | Duplicate detection sensitivity |
| `max_per_account` | agent.py | 9 | Downloads before account rotation |
| `timeout` | mcp_server.py | 90000 | Download timeout (ms) |
| `cooldown` | mcp_server.py | 40s | Wait between downloads |

---

## 8. Dependencies

### 8.1 requirements.txt

```txt
playwright>=1.40.0      # Browser automation
openai>=1.0.0           # GPT-4o and embeddings
numpy>=1.24.0           # Cosine similarity calculation
python-dotenv>=1.0.0    # Environment variable loading
postgrest>=0.16.0       # Supabase REST client
mcp>=1.0.0              # Model Context Protocol (optional)
```

### 8.2 Install Commands

```bash
# Install Python packages
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### 8.3 Why PostgREST instead of Supabase?

The official `supabase` Python package uses `httpx` which has build issues on Python 3.14+. 
We use the lighter `postgrest` package directly:

```python
from postgrest import SyncPostgrestClient

client = SyncPostgrestClient(
    f"{SUPABASE_URL}/rest/v1",
    headers={
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
)
```

---

## 9. Deployment Guide

### 9.1 First-Time Setup

```bash
# 1. Clone/download the project
cd DataCollectornValidatorAgent

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 4. Create .env file
copy .env.example .env
# Edit .env with your API keys

# 5. Create accounts.json
# Add your Z-Library credentials

# 6. Initialize Supabase table (one-time)
python memory.py
# Copy the SQL and run in Supabase Dashboard

# 7. Run the agent
python agent.py
```

### 9.2 Team Setup

Each team member needs:
1. Their own `.env` file with same Supabase credentials
2. Their own `accounts.json` (can share or use separate Z-Lib accounts)
3. Same codebase

The shared Supabase database handles all coordination.

---

## 10. Troubleshooting

### 10.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Site status check failed: 503` | Z-Library is down | Wait and try again later |
| `Site status check failed: 403` | Cloudflare blocking | Use VPN or wait |
| `Site status check failed: timeout` | Network issue | Check internet connection |
| `z-bookcard elements not found` | Z-Library page structure changed | Check if site is up, update selectors |
| `Embedding generation failed` | OpenAI API issue | Check API key, rate limits |
| `Supabase query failed` | Connection/auth issue | Verify SUPABASE_URL and KEY |
| `Download timeout` | Slow connection or blocked | Increase timeout, check IP ban |
| `All accounts exhausted` | Hit daily limits | Add more accounts or wait 24h |

### 10.2 Debug Mode

Set `headless=False` in mcp_server.py to see browser:

```python
browser = await p.chromium.launch(headless=False)  # See what's happening
```

### 10.3 Screenshots on Error

The system auto-saves debug screenshots:
- `debug_search_page.png` - When z-bookcard not found
- `debug_no_books.png` - When search returns empty

### 10.4 Logs

Monitor console output for:
- `🔎 Searching: ...` - Search initiated
- `📚 Found X books` - Books detected
- `⏭️ SKIPPING: ...` - Duplicate detected
- `💾 Saved: ...` - Download successful
- `❌ Error: ...` - Something failed

---

## Appendix: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAMESH QUICK REFERENCE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RUN:           python agent.py                                             │
│                                                                             │
│  COMMANDS:      quit     - Exit                                             │
│                 status   - Check remaining downloads                        │
│                 memory   - View shared memory stats                         │
│                                                                             │
│  FILES:         agent.py            - Main entry point                      │
│                 memory.py           - Supabase cloud memory                 │
│                 mcp_server.py       - Download engine                       │
│                 check_site_status.py - Site availability check              │
│                 accounts.json       - Z-Library credentials                 │
│                                                                             │
│  ENV VARS:      OPENAI_API_KEY - Required                                   │
│                 SUPABASE_URL   - Required                                   │
│                 SUPABASE_KEY   - Required                                   │
│                                                                             │
│  LIMITS:        9 books/account/day                                         │
│                 40 second cooldown between downloads                        │
│                 85% similarity = duplicate                                  │
│                                                                             │
│  OPENAI:        gpt-4o              - Agent brain                           │
│                 gpt-4o-mini         - Metadata cleaning                     │
│                 text-embedding-3-small - Duplicate detection                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

**Made with ❤️ and ☕ by Sajak**
