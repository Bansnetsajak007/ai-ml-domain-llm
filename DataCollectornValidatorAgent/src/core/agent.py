import asyncio
import json
import os
import sys
from openai import OpenAI
import dotenv
from enum import Enum

# Load config from config folder
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', '.env')
dotenv.load_dotenv(config_path)

# Environment validation
required_env_vars = ['OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"❌ ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("\nPlease create config/.env file with:")
    print("  OPENAI_API_KEY=your-key")
    print("  SUPABASE_URL=your-supabase-url")
    print("  SUPABASE_KEY=your-supabase-key")
    sys.exit(1)

try:
    client = OpenAI()
except Exception as e:
    print(f"❌ ERROR: Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Import the core download functions and memory system
from src.collectors.mcp_server import core_download_logic
from src.collectors.arxiv_collector import core_arxiv_download_logic, ArxivCategory, get_common_categories
from src.core.memory import AgentMemory


class DataSource(Enum):
    """Available data sources for collection."""
    ZLIBRARY = "zlibrary"
    ARXIV = "arxiv"


# Configuration
MAX_DOWNLOADS_PER_ACCOUNT = int(os.getenv("MAX_DOWNLOADS_PER_ACCOUNT", "9"))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
MAX_RETRIES = 3

# --- RAMESH ASCII BANNER ---
RAMESH_BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗  █████╗ ███╗   ███╗███████╗███████╗██╗  ██╗                         ║
║   ██╔══██╗██╔══██╗████╗ ████║██╔════╝██╔════╝██║  ██║                         ║
║   ██████╔╝███████║██╔████╔██║█████╗  ███████╗███████║                         ║
║   ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║                         ║
║   ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗███████║██║  ██║                         ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝                         ║
║                                                                               ║
║             🤖 AI-Powered Data Collector Agent v2.0                           ║
║                  📚 Z-Library Books + 📄 arXiv Papers                         ║
║                       Created by Sajak 🇳🇵                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# --- AGENT SYSTEM PROMPTS ---
ZLIBRARY_SYSTEM_PROMPT = """You are RAMESH, an AI-powered Data Collector Agent created by Sajak. You have a friendly, helpful personality with a touch of Nepali humor.

Your job is to help users download relevant books from Z-Library to build their training datasets.

## Your Capabilities:
You have access to the `download_books` tool that can search and download books on any topic.

## Your Responsibilities:
1. **Understand User Intent**: Parse what kind of dataset they're building
2. **Strategic Planning**: Break broad requests into specific, searchable topics
3. **Smart Searching**: Choose precise search terms that will yield relevant results
4. **Resource Management**: You have multiple accounts with 9 downloads each. Use the `check_remaining_downloads` tool to see your TOTAL available downloads across all accounts. The system automatically rotates accounts when one is exhausted.
5. **Report Progress**: Tell the user what you found and downloaded
6. **Suggest More**: Recommend related topics they might want

## CRITICAL - Search Term Guidelines:
- ALWAYS use specific, technical search terms to avoid irrelevant results
- For programming libraries, ALWAYS include "Python" or the language name
  - BAD: "Pandas" (will find animal books!)
  - GOOD: "Python Pandas data analysis"
- For technical topics, add context words:
  - BAD: "Transformers" (will find movie/toy books!)
  - GOOD: "Transformer deep learning NLP"
- Include keywords like: "programming", "tutorial", "handbook", "guide", "machine learning"

## Guidelines:
- Break broad topics into 2-4 specific subtopics for better search results
- Allocate downloads strategically across subtopics (e.g., 3 books each for 3 topics)
- Avoid duplicate or overlapping searches
- Prefer foundational/comprehensive books over niche ones for dataset building
- Always confirm the plan with the user before downloading

## Example Interaction:
User: "I need books on Pandas, Numpy and ML math"
You: "Let me first check how many downloads we have available..." (call check_remaining_downloads)
Then: "Great news! We have 63 downloads available across all accounts. Here's my plan:
1. 'Python Pandas data analysis tutorial' - 10 books
2. 'Python NumPy scientific computing' - 10 books  
3. 'Mathematics for machine learning' - 10 books

Total: 30 books across 3 focused areas. Should I proceed?"

Remember: Think step-by-step, be strategic, use SPECIFIC search terms, and maximize the value of each download. Always check remaining downloads first to plan properly.

## Your Personality:
- You're friendly and helpful, like a Nepali dai (elder brother)
- Use phrases like "Namaste!", "No problem, bro!", "Ramesh always delivers!"
- When waiting, mention taking a "chai break" ☕
- Be enthusiastic but professional
- Celebrate successes with the user"""

ARXIV_SYSTEM_PROMPT = """You are RAMESH, an AI-powered Data Collector Agent created by Sajak. You have a friendly, helpful personality with a touch of Nepali humor.

Your job is to help users download research papers from arXiv to build their training datasets.

## Your Capabilities:
You have access to the `download_papers` tool that can search and download papers from arXiv on any topic.

## arXiv Mode - Key Information:
- arXiv is FREE and has NO download limits! 🎉
- You can download thousands of papers (target: 4k-5k papers)
- Papers are automatically filtered to avoid duplicates
- Rate limiting is handled automatically (be patient with large downloads)

## Your Responsibilities:
1. **Understand User Intent**: Parse what kind of research papers they need
2. **Strategic Planning**: Break broad requests into specific research areas
3. **Smart Searching**: Use academic/technical search terms
4. **Category Awareness**: Suggest relevant arXiv categories for better results
5. **Report Progress**: Tell the user what you found and downloaded
6. **Bulk Operations**: For large datasets, plan multi-topic collection

## arXiv Categories (Common ML/AI):
- cs.AI: Artificial Intelligence
- cs.CL: Computation and Language (NLP)
- cs.CV: Computer Vision
- cs.LG: Machine Learning
- cs.NE: Neural and Evolutionary Computing
- stat.ML: Statistics - Machine Learning
- cs.RO: Robotics
- cs.IR: Information Retrieval

## CRITICAL - Search Guidelines:
- Use academic terminology, not casual language
- arXiv search is different from Google - be specific
- Good: "transformer attention mechanism", "large language model"
- Bad: "how transformers work", "LLM stuff"
- Include author names if looking for specific work
- Use category filters for more focused results

## Example Interaction for Large Dataset:
User: "I need 5000 papers on NLP and transformers"
You: "Namaste! Let me plan a bulk collection strategy:

1. 'transformer architecture' (cs.CL, cs.LG) - 1000 papers
2. 'language model pretraining' (cs.CL) - 1000 papers
3. 'attention mechanism neural network' (cs.LG) - 1000 papers
4. 'text generation natural language' (cs.CL) - 1000 papers
5. 'machine translation neural' (cs.CL) - 1000 papers

This will give diverse coverage of the field. Should I start?"

## Guidelines:
- For bulk collection (4k-5k papers), divide into 5-10 subtopics
- Each subtopic should get 500-1000 papers
- Suggest related research areas the user might have missed
- Always confirm the plan before starting large downloads

## Your Personality:
- You're friendly and helpful, like a Nepali dai (elder brother)
- Use phrases like "Namaste!", "No problem, bro!", "Ramesh always delivers!"
- When downloading large batches, say "Time for a long chai break! ☕"
- Be enthusiastic about research and learning
- Celebrate milestones during bulk downloads"""

# --- TOOL DEFINITIONS ---
# Z-Library Tools
ZLIBRARY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "download_books",
            "description": "Search Z-Library and download books on a specific topic. The system automatically checks memory to avoid downloading duplicates that other users already have. The system also automatically rotates through multiple accounts, so you can download many more than 9 books total. Returns the number of books successfully downloaded.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The search query/topic to find books about. Be specific for better results."
                    },
                    "max_books": {
                        "type": "integer",
                        "description": "Maximum number of books to download for this topic. Can be higher than 9 as the system rotates accounts automatically. Default is 5.",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_remaining_downloads",
            "description": "Check how many downloads are remaining for the current session across ALL accounts. Always call this first to know your total capacity before planning downloads.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_downloaded_books",
            "description": "List all books that have been downloaded in this session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search the shared memory to find what books have already been downloaded by all users. Use this to avoid duplicates and see what's already in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find similar books in memory."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_stats",
            "description": "Get statistics about the shared book memory - total books, books per user, top topics.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# arXiv Tools
ARXIV_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "download_papers",
            "description": "Search arXiv and download research papers on a specific topic. arXiv is free and has no download limits! The system automatically handles rate limiting and duplicate detection. Use this for bulk paper collection (4k-5k papers is achievable).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query/topic to find papers about. Use academic terminology for best results."
                    },
                    "max_papers": {
                        "type": "integer",
                        "description": "Maximum number of papers to download. Can be large (1000+) since arXiv has no limits. Default is 100.",
                        "default": 100
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of arXiv category codes to filter by. Examples: ['cs.CL', 'cs.LG', 'cs.AI']. If not provided, searches all categories."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_arxiv_categories",
            "description": "List common arXiv categories for ML/AI research with descriptions. Use this to help users choose appropriate category filters.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_downloaded_papers",
            "description": "List all papers that have been downloaded in this session with their arXiv IDs and titles.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_paper_memory",
            "description": "Search the shared memory to find what papers have already been downloaded by all users. Use this to avoid duplicates and see what research is already in the dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find similar papers in memory."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_memory_stats",
            "description": "Get statistics about the shared paper memory - total papers, papers per user, top arXiv categories.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Combined tools getter
def get_tools_for_source(source: DataSource) -> list:
    """Get the appropriate tools based on data source."""
    if source == DataSource.ZLIBRARY:
        return ZLIBRARY_TOOLS
    elif source == DataSource.ARXIV:
        return ARXIV_TOOLS
    return ZLIBRARY_TOOLS  # Default

def get_system_prompt_for_source(source: DataSource) -> str:
    """Get the appropriate system prompt based on data source."""
    if source == DataSource.ZLIBRARY:
        return ZLIBRARY_SYSTEM_PROMPT
    elif source == DataSource.ARXIV:
        return ARXIV_SYSTEM_PROMPT
    return ZLIBRARY_SYSTEM_PROMPT  # Default


class LibrarianAgent:
    def __init__(self, user_name: str = "default_user", source: DataSource = DataSource.ZLIBRARY):
        self.user_name = user_name
        self.source = source
        self.accounts = self._load_accounts() if source == DataSource.ZLIBRARY else []
        self.current_account_idx = 0
        self.downloads_on_current_account = 0
        self.max_per_account = MAX_DOWNLOADS_PER_ACCOUNT
        self.session_downloads = []  # Track books downloaded
        self.session_papers = []     # Track papers downloaded
        
        # Set up conversation with appropriate system prompt
        system_prompt = get_system_prompt_for_source(source)
        self.conversation = [{"role": "system", "content": system_prompt}]
        
        # Initialize memory with proper error handling
        try:
            self.memory = AgentMemory()  # Shared memory system
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize shared memory: {e}")
            print("   Continuing without duplicate detection...")
            self.memory = None
    
    def _load_accounts(self):
        """Load accounts from ZLIB_ACCOUNTS env var or fallback to accounts.json"""
        # Only needed for Z-Library mode
        if self.source != DataSource.ZLIBRARY:
            return []
        
        try:
            accounts_json = os.getenv("ZLIB_ACCOUNTS")
            if accounts_json:
                accounts = json.loads(accounts_json)
            elif os.path.exists("accounts.json"):
                with open("accounts.json", "r") as f:
                    accounts = json.load(f)
            else:
                print("⚠️ Warning: No Z-Library accounts found. Z-Library mode will not work.")
                print("   Set ZLIB_ACCOUNTS env var or create accounts.json")
                return []
            
            if not accounts:
                print("⚠️ Warning: No accounts in configuration")
                return []
            return accounts
        except json.JSONDecodeError:
            print("❌ ERROR: Accounts JSON is invalid")
            return []
    
    def _trim_conversation_history(self):
        """Trim conversation history to prevent memory leak and token overflow."""
        # Keep system prompt + last N messages
        if len(self.conversation) > MAX_CONVERSATION_HISTORY:
            system_msg = self.conversation[0]
            recent_msgs = self.conversation[-(MAX_CONVERSATION_HISTORY-1):]
            self.conversation = [system_msg] + recent_msgs
            print(f"\n💡 Trimmed conversation history to last {MAX_CONVERSATION_HISTORY} messages")
    
    @property
    def current_account(self):
        if self.current_account_idx < len(self.accounts):
            return self.accounts[self.current_account_idx]
        return None
    
    @property
    def remaining_downloads(self):
        total_remaining = 0
        # Remaining on current account
        total_remaining += self.max_per_account - self.downloads_on_current_account
        # Plus all remaining accounts
        remaining_accounts = len(self.accounts) - self.current_account_idx - 1
        total_remaining += remaining_accounts * self.max_per_account
        return total_remaining
    
    def _rotate_account_if_needed(self):
        """Switch to next account if current one is exhausted."""
        if self.downloads_on_current_account >= self.max_per_account:
            self.current_account_idx += 1
            self.downloads_on_current_account = 0
            
            if self.current_account_idx < len(self.accounts):
                print(f"\n🔄 Account limit reached! No worries, Ramesh has backup!")
                print(f"   Switching to {self.accounts[self.current_account_idx]['name']}...")
                return True
            else:
                print("\n🛑 All accounts exhausted! Ramesh tried his best! 🙏")
                return False
        return True
    
    async def execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """Execute a tool and return the result as a string."""
        
        # =====================
        # Z-LIBRARY TOOLS
        # =====================
        if tool_name == "download_books":
            topic = tool_args.get("topic")
            requested_books = tool_args.get("max_books", 5)
            total_downloaded = 0
            all_downloaded_books = []
            
            print(f"\n🔧 Ramesh is working: download_books(topic='{topic}', max_books={requested_books})")
            print(f"   👤 User: {self.user_name}")
            
            # Loop through accounts until we've downloaded all requested books
            while total_downloaded < requested_books:
                # Check if current account is exhausted, rotate if needed
                if self.downloads_on_current_account >= self.max_per_account:
                    if not self._rotate_account_if_needed():
                        break  # All accounts exhausted
                
                if not self.current_account:
                    break
                
                # Calculate how many we can download from current account
                remaining_on_account = self.max_per_account - self.downloads_on_current_account
                books_to_download = min(remaining_on_account, requested_books - total_downloaded)
                
                if books_to_download <= 0:
                    break
                
                print(f"\n   📊 Using account: {self.current_account['name']}")
                print(f"   📊 Downloads remaining on this account: {remaining_on_account}")
                print(f"   📥 Downloading {books_to_download} books in this batch...")
                
                try:
                    result, count, downloaded_books = await core_download_logic(
                        topic=topic,
                        account=self.current_account,
                        max_books=books_to_download,
                        memory=self.memory,
                        user_name=self.user_name
                    )
                    
                    # --- HANDLE LIMIT_REACHED SIGNAL ---
                    # Z-Library showed "Daily limit reached" page - force rotation
                    if result == "LIMIT_REACHED":
                        print(f"\n🚫 Account {self.current_account['name']} hit daily limit!")
                        # Force this account as exhausted
                        self.downloads_on_current_account = self.max_per_account
                        if not self._rotate_account_if_needed():
                            break  # All accounts exhausted
                        # Continue loop to try with next account
                        continue
                    
                    # Check if there was an error (site down, etc.)
                    if count == 0 and result.startswith("Site status check failed"):
                        return f"ERROR: {result}. The site might be down or blocking requests. Try again later."
                    
                    # Update counters
                    self.downloads_on_current_account += count
                    total_downloaded += count
                    all_downloaded_books.extend(downloaded_books)
                    
                    # If no books found for this topic, don't keep trying
                    if count == 0:
                        break
                    
                    # If we got fewer than requested, no more books available for this topic
                    if count < books_to_download:
                        break
                        
                except Exception as e:
                    return f"ERROR downloading books: {str(e)}"
            
            # Record session downloads
            if total_downloaded > 0:
                self.session_downloads.append({
                    "topic": topic,
                    "count": total_downloaded,
                    "account": "multiple",
                    "books": all_downloaded_books
                })
            
            if total_downloaded == 0:
                return f"No books found for '{topic}'. Try different search terms. Remaining downloads: {self.remaining_downloads}."
            
            return f"Successfully downloaded {total_downloaded} books about '{topic}'. Total session downloads: {sum(d['count'] for d in self.session_downloads)}. Remaining downloads: {self.remaining_downloads}."
        
        elif tool_name == "check_remaining_downloads":
            return f"Remaining downloads: {self.remaining_downloads} (Current account: {self.current_account['name'] if self.current_account else 'None'}, {self.max_per_account - self.downloads_on_current_account} left on this account)"
        
        elif tool_name == "list_downloaded_books":
            if not self.session_downloads:
                return "No books downloaded yet in this session."
            
            summary = "Books downloaded this session:\n"
            for d in self.session_downloads:
                summary += f"  - {d['topic']}: {d['count']} books (via {d['account']})\n"
            summary += f"\nTotal: {sum(d['count'] for d in self.session_downloads)} books"
            return summary
        
        elif tool_name == "search_memory":
            query = tool_args.get("query", "")
            results = self.memory.search_similar(query, limit=10)
            
            if not results:
                return f"No books found in memory matching '{query}'."
            
            summary = f"Books in memory similar to '{query}':\n"
            for book in results:
                summary += f"  - {book['title']} by {book['authors']} (downloaded by {book['downloaded_by']}, similarity: {book['similarity']:.2f})\n"
            return summary
        
        elif tool_name == "get_memory_stats":
            stats = self.memory.get_stats()
            summary = f"📚 Memory Statistics:\n"
            summary += f"  Total books in dataset: {stats['total_books']}\n"
            summary += f"  Books by user:\n"
            for user, count in stats['by_user'].items():
                summary += f"    - {user}: {count} books\n"
            summary += f"  Top topics:\n"
            for topic, count in list(stats['top_topics'].items())[:5]:
                summary += f"    - {topic}: {count} books\n"
            return summary
        
        # =====================
        # ARXIV TOOLS
        # =====================
        elif tool_name == "download_papers":
            query = tool_args.get("query")
            max_papers = tool_args.get("max_papers", 100)
            categories = tool_args.get("categories")
            
            print(f"\n🔧 Ramesh is working: download_papers(query='{query}', max_papers={max_papers})")
            print(f"   👤 User: {self.user_name}")
            if categories:
                print(f"   🏷️ Categories: {categories}")
            
            try:
                message, count, papers = await core_arxiv_download_logic(
                    query=query,
                    max_papers=max_papers,
                    categories=categories,
                    memory=self.memory,
                    user_name=self.user_name
                )
                
                # Record session papers
                if count > 0:
                    self.session_papers.append({
                        "query": query,
                        "categories": categories,
                        "count": count,
                        "papers": papers
                    })
                
                total_session_papers = sum(p['count'] for p in self.session_papers)
                
                if count == 0:
                    return f"No papers found for '{query}'. Try different search terms or categories."
                
                return f"Successfully downloaded {count} papers about '{query}'. Total session papers: {total_session_papers}."
                
            except Exception as e:
                return f"ERROR downloading papers: {str(e)}"
        
        elif tool_name == "list_arxiv_categories":
            categories_info = """Common arXiv Categories for ML/AI Research:

🤖 Artificial Intelligence & Machine Learning:
  - cs.AI: Artificial Intelligence
  - cs.LG: Machine Learning (main ML category)
  - stat.ML: Statistics - Machine Learning
  - cs.NE: Neural and Evolutionary Computing

📝 Natural Language Processing:
  - cs.CL: Computation and Language (NLP)
  - cs.IR: Information Retrieval

👁️ Computer Vision:
  - cs.CV: Computer Vision and Pattern Recognition

🤖 Robotics & Control:
  - cs.RO: Robotics
  - cs.SY: Systems and Control

💻 Other CS Categories:
  - cs.DC: Distributed Computing
  - cs.CR: Cryptography and Security
  - cs.DB: Databases
  - cs.SE: Software Engineering
  - cs.PL: Programming Languages

🔬 Related Fields:
  - eess.SP: Signal Processing
  - quant-ph: Quantum Physics
  - math.OC: Optimization and Control

Use these category codes in the 'categories' parameter to filter results."""
            return categories_info
        
        elif tool_name == "list_downloaded_papers":
            if not self.session_papers:
                return "No papers downloaded yet in this session."
            
            summary = "Papers downloaded this session:\n"
            for p in self.session_papers:
                cat_str = f" [{', '.join(p['categories'])}]" if p['categories'] else ""
                summary += f"  - '{p['query']}'{cat_str}: {p['count']} papers\n"
            summary += f"\nTotal: {sum(p['count'] for p in self.session_papers)} papers"
            return summary
        
        elif tool_name == "search_paper_memory":
            query = tool_args.get("query", "")
            if not self.memory:
                return "Memory system not available."
            
            results = self.memory.search_papers_similar(query, limit=10)
            
            if not results:
                return f"No papers found in memory matching '{query}'."
            
            summary = f"Papers in memory similar to '{query}':\n"
            for paper in results:
                summary += f"  - [{paper['arxiv_id']}] {paper['title'][:60]}... (by {paper['downloaded_by']}, similarity: {paper['similarity']:.2f})\n"
            return summary
        
        elif tool_name == "get_paper_memory_stats":
            if not self.memory:
                return "Memory system not available."
            
            stats = self.memory.get_paper_stats()
            summary = f"📄 Paper Memory Statistics:\n"
            summary += f"  Total papers in dataset: {stats['total_papers']}\n"
            summary += f"  Papers by user:\n"
            for user, count in stats['by_user'].items():
                summary += f"    - {user}: {count} papers\n"
            summary += f"  Top categories:\n"
            for cat, count in list(stats['top_categories'].items())[:5]:
                summary += f"    - {cat}: {count} papers\n"
            return summary
        
        else:
            return f"Unknown tool: {tool_name}"
    
    async def chat(self, user_message: str) -> str:
        """Send a message to the agent and get a response."""
        
        self.conversation.append({"role": "user", "content": user_message})
        
        # Trim history to prevent memory leak
        self._trim_conversation_history()
        
        # Get appropriate tools for current source
        tools = get_tools_for_source(self.source)
        
        while True:  # Loop to handle multiple tool calls
            # Retry logic for LLM calls
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",  # Use gpt-4o for best reasoning
                        messages=self.conversation,
                        tools=tools,
                        tool_choice="auto",
                        timeout=30
                    )
                    break  # Success
                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        print(f"\n⚠️ LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        return f"❌ Sorry, I'm having trouble connecting to my brain right now. Error: {str(last_error)}"
            
            message = response.choices[0].message
            
            # Check if the agent wants to call tools
            if message.tool_calls:
                # Add assistant message with tool calls
                self.conversation.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the tool
                    result = await self.execute_tool(func_name, func_args)
                    
                    # Add tool result to conversation
                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                # Continue the loop to let the agent process tool results
                continue
            
            else:
                # No tool calls, just a text response
                self.conversation.append({"role": "assistant", "content": message.content})
                return message.content
    
    def reset_session(self):
        """Reset for a new session (keeps conversation history)."""
        self.current_account_idx = 0
        self.downloads_on_current_account = 0
        self.session_downloads = []
        self.session_papers = []
    
    def switch_source(self, new_source: DataSource):
        """Switch between Z-Library and arXiv modes."""
        self.source = new_source
        system_prompt = get_system_prompt_for_source(new_source)
        self.conversation = [{"role": "system", "content": system_prompt}]
        self.reset_session()
        
        # Reload accounts if switching to Z-Library
        if new_source == DataSource.ZLIBRARY:
            self.accounts = self._load_accounts()
        
        print(f"\n🔄 Switched to {new_source.value.upper()} mode!")


async def main():
    print(RAMESH_BANNER)
    
    # Ask for user identification
    print("🙏 Namaste! I'm Ramesh, your AI-powered data collector.")
    print("   Now with DUAL MODE: Books from Z-Library + Papers from arXiv!")
    print("")
    print("👤 Who are you? (This helps me track who downloaded what)")
    user_name_input = input("   Your name: ").strip()
    
    # Sanitize user input
    user_name = "".join(c for c in user_name_input if c.isalnum() or c in (' ', '-', '_'))[:50]
    if not user_name:
        user_name = "anonymous"
    
    # Ask for data source
    print("")
    print("═"*73)
    print("📦 SELECT DATA SOURCE:")
    print("═"*73)
    print("")
    print("   [1] 📚 Z-Library (Books)")
    print("       - Download textbooks, technical books, guides")
    print("       - Limited by account quotas (but we have multiple accounts!)")
    print("       - Requires browser automation")
    print("")
    print("   [2] 📄 arXiv (Research Papers)")
    print("       - Download research papers, preprints")
    print("       - NO download limits! Perfect for large datasets (4k-5k papers)")
    print("       - Fast API-based collection")
    print("")
    
    while True:
        choice = input("   Enter 1 or 2 (default: 2 for arXiv): ").strip()
        if choice in ['', '2']:
            selected_source = DataSource.ARXIV
            break
        elif choice == '1':
            selected_source = DataSource.ZLIBRARY
            break
        else:
            print("   ⚠️ Please enter 1 or 2")
    
    print(f"\n✅ Selected: {selected_source.value.upper()} mode")
    
    # Create agent with selected source
    agent = LibrarianAgent(user_name=user_name, source=selected_source)
    
    # Show memory stats (with error handling)
    try:
        if agent.memory:
            combined_stats = agent.memory.get_combined_stats()
            book_stats = combined_stats.get("books", {"total_books": 0, "by_user": {}})
            paper_stats = combined_stats.get("papers", {"total_papers": 0, "by_user": {}})
        else:
            book_stats = {"total_books": 0, "by_user": {}}
            paper_stats = {"total_papers": 0, "by_user": {}}
    except Exception as e:
        print(f"⚠️ Could not retrieve memory stats: {e}")
        book_stats = {"total_books": 0, "by_user": {}}
        paper_stats = {"total_papers": 0, "by_user": {}}
    
    name_lower = user_name.lower()
    if name_lower == "sajak":
        greeting_suffix = "dai"  # malai matra dai vanxa hahahahahahahaha
    elif name_lower == "dipsan":
        greeting_suffix = "didi"
    elif name_lower == "siddarth":
        greeting_suffix = "muji"
    elif name_lower == "ronish":
        greeting_suffix = "Please"
    else:
        greeting_suffix = "dai/didi"  # Default for others
    
    print(f"\n🙏 Welcome, {user_name} {greeting_suffix}!")
    print(f"")
    print(f"┌─────────────────────────────────────────────────────────────────┐")
    print(f"│  📊 STATUS                                                      │")
    print(f"├─────────────────────────────────────────────────────────────────┤")
    print(f"│  🎯 Mode: {selected_source.value.upper():<53} │")
    
    if selected_source == DataSource.ZLIBRARY:
        print(f"│  🔑 Accounts loaded: {len(agent.accounts):<41} │")
        print(f"│  📥 Max downloads available: {agent.remaining_downloads:<33} │")
    else:
        print(f"│  ♾️  arXiv has NO download limits!                              │")
        print(f"│  🚀 Target: 4,000 - 5,000 papers                               │")
    
    print(f"│  📚 Books in memory: {book_stats.get('total_books', 0):<41} │")
    print(f"│  📄 Papers in memory: {paper_stats.get('total_papers', 0):<40} │")
    print(f"└─────────────────────────────────────────────────────────────────┘")
    
    if book_stats.get('by_user'):
        print(f"\n📚 Books downloaded by team:")
        for user, count in book_stats['by_user'].items():
            print(f"      - {user}: {count} books")
    
    if paper_stats.get('by_user'):
        print(f"\n📄 Papers downloaded by team:")
        for user, count in paper_stats['by_user'].items():
            print(f"      - {user}: {count} papers")
    
    if selected_source == DataSource.ARXIV:
        print("\n💬 Ready to collect research papers!")
        print("   Examples:")
        print("   - 'I need 1000 papers on transformer architectures'")
        print("   - 'Get 500 papers about reinforcement learning from cs.LG'")
        print("   - 'Collect 5000 papers on NLP and machine translation'")
    else:
        print("\n💬 Bhannus ta, what kind of books do you need?")
        print("   Examples:")
        print("   - 'I need books about machine learning mathematics'")
        print("   - 'Build me a dataset for learning NLP and transformers'")
        print("   - 'Get books on statistics, probability, and linear algebra'")
    
    print("")
    print("   Commands:")
    print("   - 'quit' - Exit Ramesh")
    print("   - 'status' - Show current session status")
    print("   - 'memory' - Show shared memory statistics")
    print("   - 'switch' - Switch between Z-Library and arXiv modes")
    print("")
    print("═"*73)
    
    while True:
        try:
            mode_indicator = "📄" if agent.source == DataSource.ARXIV else "📚"
            user_input = input(f"\n{mode_indicator} {user_name}: ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n🙏 Dhanyabad! Thank you for using Ramesh!")
            print("   See you next time, bro! Happy learning! 📚📄")
            break
        
        if user_input.lower() == 'status':
            print(f"\n📊 SESSION STATUS:")
            print(f"   Mode: {agent.source.value.upper()}")
            
            if agent.source == DataSource.ZLIBRARY:
                print(f"   Downloads remaining: {agent.remaining_downloads}")
                print(f"   Current account: {agent.current_account['name'] if agent.current_account else 'None'}")
                print(f"   Books this session: {sum(d['count'] for d in agent.session_downloads)}")
            else:
                print(f"   Papers this session: {sum(p['count'] for p in agent.session_papers)}")
                if agent.session_papers:
                    print(f"   Topics searched:")
                    for p in agent.session_papers:
                        print(f"      - {p['query']}: {p['count']} papers")
            continue
        
        if user_input.lower() == 'memory':
            try:
                combined = agent.memory.get_combined_stats() if agent.memory else {}
                book_stats = combined.get('books', {"total_books": 0, "by_user": {}, "top_topics": {}})
                paper_stats = combined.get('papers', {"total_papers": 0, "by_user": {}, "top_categories": {}})
                
                print(f"\n📊 SHARED MEMORY STATS:")
                print(f"\n📚 Books:")
                print(f"   Total: {book_stats.get('total_books', 0)}")
                for user, count in book_stats.get('by_user', {}).items():
                    print(f"      - {user}: {count}")
                
                print(f"\n📄 Papers:")
                print(f"   Total: {paper_stats.get('total_papers', 0)}")
                for user, count in paper_stats.get('by_user', {}).items():
                    print(f"      - {user}: {count}")
                
                if paper_stats.get('top_categories'):
                    print(f"\n🏷️ Top arXiv categories:")
                    for cat, count in list(paper_stats['top_categories'].items())[:5]:
                        print(f"      - {cat}: {count}")
            except Exception as e:
                print(f"⚠️ Could not fetch memory stats: {e}")
            continue
        
        if user_input.lower() == 'switch':
            if agent.source == DataSource.ZLIBRARY:
                agent.switch_source(DataSource.ARXIV)
                print("   Now collecting from arXiv (Research Papers)")
            else:
                agent.switch_source(DataSource.ZLIBRARY)
                print("   Now collecting from Z-Library (Books)")
            continue
        
        print("\n🤔 Ramesh is thinking... (sipping chai ☕)\n")
        
        try:
            response = await agent.chat(user_input)
            print(f"\n🤖 Ramesh: {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
