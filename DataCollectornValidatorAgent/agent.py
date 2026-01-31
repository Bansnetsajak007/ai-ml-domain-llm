import asyncio
import json
import os
import sys
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# Environment validation
required_env_vars = ['OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"❌ ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("\nPlease create a .env file with:")
    print("  OPENAI_API_KEY=your-key")
    print("  SUPABASE_URL=your-supabase-url")
    print("  SUPABASE_KEY=your-supabase-key")
    sys.exit(1)

try:
    client = OpenAI()
except Exception as e:
    print(f"❌ ERROR: Failed to initialize OpenAI client: {e}")
    sys.exit(1)

# Import the core download function and memory system
from mcp_server import core_download_logic
from memory import AgentMemory

# Configuration
MAX_DOWNLOADS_PER_ACCOUNT = int(os.getenv("MAX_DOWNLOADS_PER_ACCOUNT", "9"))
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))  # Prevent memory leak
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
║                  🤖 AI-Powered Data Collector Agent                           ║
║                       Created by Sajak 🇳🇵                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# --- AGENT SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are RAMESH, an AI-powered Data Collector Agent created by Sajak. You have a friendly, helpful personality with a touch of Nepali humor.

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

# --- TOOL DEFINITIONS ---
TOOLS = [
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


class LibrarianAgent:
    def __init__(self, user_name: str = "default_user"):
        self.user_name = user_name
        self.accounts = self._load_accounts()
        self.current_account_idx = 0
        self.downloads_on_current_account = 0
        self.max_per_account = MAX_DOWNLOADS_PER_ACCOUNT
        self.session_downloads = []  # Track what we've downloaded
        self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Initialize memory with proper error handling
        try:
            self.memory = AgentMemory()  # Shared memory system
        except Exception as e:
            print(f"⚠️ Warning: Failed to initialize shared memory: {e}")
            print("   Continuing without duplicate detection...")
            self.memory = None
    
    def _load_accounts(self):
        """Load accounts from ZLIB_ACCOUNTS env var or fallback to accounts.json"""
        try:
            accounts_json = os.getenv("ZLIB_ACCOUNTS")
            if accounts_json:
                accounts = json.loads(accounts_json)
            elif os.path.exists("accounts.json"):
                with open("accounts.json", "r") as f:
                    accounts = json.load(f)
            else:
                print("❌ ERROR: No accounts found! Set ZLIB_ACCOUNTS env var or create accounts.json")
                sys.exit(1)
            
            if not accounts:
                raise ValueError("No accounts found")
            return accounts
        except json.JSONDecodeError:
            print("❌ ERROR: Accounts JSON is invalid")
            sys.exit(1)
    
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
        
        else:
            return f"Unknown tool: {tool_name}"
    
    async def chat(self, user_message: str) -> str:
        """Send a message to the agent and get a response."""
        
        self.conversation.append({"role": "user", "content": user_message})
        
        # Trim history to prevent memory leak
        self._trim_conversation_history()
        
        while True:  # Loop to handle multiple tool calls
            # Retry logic for LLM calls
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",  # Use gpt-4o for best reasoning
                        messages=self.conversation,
                        tools=TOOLS,
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


async def main():
    print(RAMESH_BANNER)
    
    # Ask for user identification
    print("🙏 Namaste! I'm Ramesh, your AI-powered data collector.")
    print("")
    print("👤 Who are you? (This helps me track who downloaded what)")
    user_name_input = input("   Your name: ").strip()
    
    # Sanitize user input
    user_name = "".join(c for c in user_name_input if c.isalnum() or c in (' ', '-', '_'))[:50]
    if not user_name:
        user_name = "anonymous"
    
    agent = LibrarianAgent(user_name=user_name)
    
    # Show memory stats (with error handling)
    try:
        stats = agent.memory.get_stats() if agent.memory else {"total_books": 0, "by_user": {}}
    except Exception as e:
        print(f"⚠️ Could not retrieve memory stats: {e}")
        stats = {"total_books": 0, "by_user": {}}
    
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
    print(f"│  🔑 Accounts loaded: {len(agent.accounts):<41} │")
    print(f"│  📥 Max downloads available: {agent.remaining_downloads:<33} │")
    print(f"│  📚 Books in shared memory: {stats['total_books']:<34} │")
    print(f"└─────────────────────────────────────────────────────────────────┘")
    
    if stats['by_user']:
        print(f"\n📚 Downloaded by team:")
        for user, count in stats['by_user'].items():
            print(f"      - {user}: {count} books")
    
    print("\n💬 Bhannus ta, what kind of dataset do you need?")
    print("   (Tell me what you need, I'll handle the rest!)")
    print("")
    print("   Examples:")
    print("   - 'I need books about machine learning mathematics'")
    print("   - 'Build me a dataset for learning NLP and transformers'")
    print("   - 'Get books on statistics, probability, and linear algebra'")
    print("")
    print("   Commands: 'quit', 'status', 'memory'")
    print("")
    print("═"*73)
    
    while True:
        try:
            user_input = input(f"\n📝 {user_name}: ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n� Dhanyabad! Thank you for using Ramesh!")
            print("   See you next time, bro! Happy learning! 📚")
            break
        
        if user_input.lower() == 'status':
            print(f"\n📊 Downloads remaining: {agent.remaining_downloads}")
            print(f"📊 Current account: {agent.current_account['name'] if agent.current_account else 'None'}")
            print(f"📊 Books this session: {sum(d['count'] for d in agent.session_downloads)}")
            continue
        
        if user_input.lower() == 'memory':
            stats = agent.memory.get_stats()
            print(f"\n📚 Shared Memory Stats:")
            print(f"   Total books: {stats['total_books']}")
            print(f"   By user:")
            for user, count in stats['by_user'].items():
                print(f"      - {user}: {count} books")
            print(f"   Top topics:")
            for topic, count in list(stats['top_topics'].items())[:5]:
                print(f"      - {topic}: {count} books")
            continue
        
        print("\n🤔 Ramesh is thinking... (sipping chai ☕)\n")
        
        try:
            response = await agent.chat(user_input)
            print(f"\n🤖 Ramesh: {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
