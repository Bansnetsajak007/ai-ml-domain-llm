import asyncio
import json
import os
import dotenv
from fastmcp import FastMCP
from playwright.async_api import async_playwright
from openai import OpenAI

# Load environment variables from config folder
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', '.env')
dotenv.load_dotenv(config_path)

# --- CONFIGURATION ---
BASE_URL = "https://z-lib.sk"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DOWNLOAD_FOLDER = os.path.abspath(os.getenv("DATA_FOLDER", os.path.join(DATA_DIR, "books")))
RESOURCES_FILE = os.getenv("RESOURCES_FILE", os.path.join(DATA_DIR, "resources.json"))
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
DOWNLOAD_COOLDOWN = int(os.getenv("DOWNLOAD_COOLDOWN", "40"))  # seconds
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))  # seconds

def load_accounts():
    """Load accounts from ZLIB_ACCOUNTS env var or fallback to accounts.json"""
    accounts_json = os.getenv("ZLIB_ACCOUNTS")
    if accounts_json:
        return json.loads(accounts_json)
    # Fallback to file for local development
    config_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'config')
    accounts_file = os.path.join(config_dir, "accounts.json")
    if os.path.exists(accounts_file):
        with open(accounts_file, "r") as f:
            return json.load(f)
    raise ValueError("No accounts found! Set ZLIB_ACCOUNTS env var or create config/accounts.json")
client = OpenAI() 

mcp = FastMCP("ExpertLibrarian")

# Ensure folders exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
if not os.path.exists(RESOURCES_FILE):
    with open(RESOURCES_FILE, "w") as f: json.dump({"resources": []}, f)

# --- LLM CLEANING WORKER ---
def clean_metadata_with_llm(raw_title):
    """
    Uses GPT-4o-mini to turn a raw filename into your structured JSON format.
    """
    prompt = f"""
    Extract metadata from this book title/filename: "{raw_title}"
    Return ONLY a JSON object with this exact schema:
    {{
        "resource_type": "Book",
        "title": "Exact Title",
        "normalized_title": "lowercase title without special chars",
        "authors": ["Author 1", "Author 2"],
        "source": "Z-Library"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Fast and cheap
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            timeout=LLM_TIMEOUT
        )
        return json.loads(response.choices[0].message.content)
    except TimeoutError:
        print(f"⚠️ LLM Cleaning timed out after {LLM_TIMEOUT}s")
        return {
            "resource_type": "Book",
            "title": raw_title,
            "normalized_title": raw_title.lower(),
            "authors": ["Unknown"],
            "source": "Z-Library"
        }
    except Exception as e:
        print(f"⚠️ LLM Cleaning failed: {e}")
        # Fallback if LLM fails
        return {
            "resource_type": "Book",
            "title": raw_title,
            "normalized_title": raw_title.lower(),
            "authors": ["Unknown"],
            "source": "Z-Library"
        }

# --- DATABASE & FILE LOGGING ---
def save_to_json_file(metadata):
    """Appends the new clean entry to resources.json"""
    try:
        with open(RESOURCES_FILE, "r+") as f:
            data = json.load(f)
            data["resources"].append(metadata)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()  # Fix: Remove any trailing data from old file
    except json.JSONDecodeError:
        # File corrupted, create fresh
        with open(RESOURCES_FILE, "w") as f:
            json.dump({"resources": [metadata]}, f, indent=4)
    except Exception as e:
        print(f"⚠️ Failed to save to JSON file: {e}")

# REMOVED: SQLite functions - using Supabase memory system exclusively
# All duplicate checking now happens through AgentMemory class

# --- 1. THE CORE LOGIC (Plain Python - Callable by agent.py) ---
async def core_download_logic(topic: str, account: dict = None, max_books: int = 9, 
                               memory=None, user_name: str = "unknown"):
    """
    The actual Playwright automation logic. 
    Args:
        topic: Search topic
        account: Single account dict with name, remix_userid, remix_userkey
        max_books: Maximum books to download this session
        memory: AgentMemory instance for duplicate checking
        user_name: Name of the user downloading (for tracking)
    Returns: 
        (message, books_downloaded, list_of_books) tuple
    """
    # If no account provided, load first from config (for MCP tool compatibility)
    if account is None:
        accounts = load_accounts()
        account = accounts[0]
    
    download_count = 0
    downloaded_books = []  # Track what we downloaded

    async with async_playwright() as p:
                # Configurable headless mode (set BROWSER_HEADLESS=true for production)
                browser = await p.chromium.launch(headless=BROWSER_HEADLESS)
                
                print(f"\n🚀 STARTING SESSION: {account['name']}")

                context = await browser.new_context()
                await context.add_cookies([
                    {'name': 'remix_userid', 'value': account['remix_userid'], 'domain': '.z-lib.sk', 'path': '/'},
                    {'name': 'remix_userkey', 'value': account['remix_userkey'], 'domain': '.z-lib.sk', 'path': '/'}
                ])
                page = await context.new_page()
                
                # --- 🛑 MANUAL INTERVENTION BLOCK 🛑 ---
                try:
                    print("Checking homepage...")
                    await page.goto(f"{BASE_URL}/", timeout=30000)
                    
                    print("\n" + "="*50)
                    print(f"🛑 MANUAL OVERRIDE: Please close the popup now!")
                    print(f"⏳ Waiting 10 seconds for you to act...")
                    print("="*50 + "\n")
                    
                    await asyncio.sleep(10) 
                    print("✅ Resuming automation...")
                        
                except Exception as e:
                    print(f"Navigation error: {e}")
                # ---------------------------------------
                
                try:
                    print(f"🔎 Searching: {topic} (PDF only)...")
                    # Filter for PDF format only using Z-Library's extension filter
                    search_url = f"{BASE_URL}/s/{topic}?extensions[]=pdf"
                    await page.goto(search_url, timeout=30000)
                    
                    # --- ROBUST WAIT LOGIC ---
                    # Wait for network idle and the custom z-bookcard elements to load
                    await page.wait_for_load_state("networkidle", timeout=30000)
                    
                    # Z-Library uses custom <z-bookcard> web components
                    # Wait for them to be ready (they get class="ready" when loaded)
                    try:
                        await page.wait_for_selector("z-bookcard.ready", timeout=15000)
                    except:
                        print(f"⚠️ z-bookcard elements not found, trying fallback...")
                        await page.screenshot(path="debug_search_page.png")
                    
                    # Get all z-bookcard elements
                    bookcards = await page.locator("z-bookcard.ready").all()
                    
                    if not bookcards:
                        print("⚠️ No book cards found on page.")
                        await page.screenshot(path="debug_no_books.png")
                    else:
                        print(f"📚 Found {len(bookcards)} books via z-bookcard elements...")
                        # -------------------------------
                        
                        processed_ids = set()  # Track by book ID to avoid duplicates
                        
                        for idx, bookcard in enumerate(bookcards):
                            if download_count >= max_books: break  # Stop when we hit the limit
                            
                            try:
                                # Extract attributes directly from z-bookcard element
                                book_id = await bookcard.get_attribute("id")
                                download_path = await bookcard.get_attribute("download")
                                book_href = await bookcard.get_attribute("href")
                                file_extension = await bookcard.get_attribute("extension") or "pdf"
                                
                                # Skip if already processed or missing download link
                                if not download_path or book_id in processed_ids:
                                    continue
                                processed_ids.add(book_id)
                                
                                # Get title and author from slot elements
                                title_el = bookcard.locator("div[slot='title']")
                                author_el = bookcard.locator("div[slot='author']")
                                
                                raw_title = await title_el.inner_text() if await title_el.count() > 0 else f"Book_{book_id}"
                                raw_author = await author_el.inner_text() if await author_el.count() > 0 else "Unknown"
                                
                                if not raw_title.strip(): 
                                    continue 
                                
                                # --- CHECK MEMORY BEFORE DOWNLOAD ---
                                if memory:
                                    dup_check = memory.check_duplicate(raw_title, raw_author)
                                    if dup_check["is_duplicate"]:
                                        similar = dup_check["similar_book"]
                                        print(f"\n⏭️ SKIPPING: {raw_title[:50]}...")
                                        print(f"   🔄 Already downloaded by {similar['downloaded_by']}")
                                        print(f"   📚 Similar to: {similar['title'][:50]}")
                                        print(f"   📊 Similarity: {dup_check['similarity']:.1%}")
                                        continue
                                
                                print(f"\n📖 [{download_count+1}/{max_books}] {raw_title[:60]}...")
                                print(f"   👤 Author: {raw_author[:40]}")
                                print(f"   📄 Format: {file_extension.upper()}")
                                
                                # Build full download URL
                                full_download_url = f"{BASE_URL}{download_path}"
                                
                                # Create clean filename from title
                                clean_title = "".join(c for c in raw_title if c.isalnum() or c in (' ', '-', '_')).strip()
                                clean_title = clean_title[:80]  # Limit filename length
                                proper_filename = f"{clean_title}.{file_extension}"
                                
                                # --- DIRECT DOWNLOAD ---
                                try:
                                    async with page.expect_download(timeout=90000) as download_info:
                                        # Use evaluate to click a link instead of goto (avoids "Download is starting" error)
                                        await page.evaluate(f"window.location.href = '{full_download_url}'")
                                    
                                    download = await download_info.value
                                    save_path = os.path.join(DOWNLOAD_FOLDER, proper_filename)
                                    await download.save_as(save_path)
                                    print(f"   💾 Saved: {proper_filename}")
                                    
                                    # LLM METADATA - include author info
                                    print(f"   🧠 LLM analyzing metadata...")
                                    clean_meta = clean_metadata_with_llm(f"{raw_title} by {raw_author}")
                                    
                                    # SAVE TO JSON
                                    save_to_json_file(clean_meta)
                                    
                                    # --- WRITE TO SHARED MEMORY (After Download) ---
                                    if memory:
                                        memory.add_book(
                                            title=clean_meta.get("title", raw_title),
                                            authors=raw_author,
                                            source="Z-Library",
                                            search_topic=topic,
                                            downloaded_by=user_name
                                        )
                                        print(f"   🧠 Added to shared memory")
                                    
                                    # Track downloaded book
                                    downloaded_books.append({
                                        "title": clean_meta.get("title", raw_title),
                                        "authors": raw_author,
                                        "filename": proper_filename
                                    })
                                    
                                    print(f"   ✅ Indexed: {clean_meta['title'][:50]}...")
                                    download_count += 1
                                    
                                    # Only cooldown if we successfully downloaded
                                    print(f"   ⏳ Cooling down for {DOWNLOAD_COOLDOWN}s...")
                                    await asyncio.sleep(DOWNLOAD_COOLDOWN)
                                    
                                    # Go back to search results (keep PDF filter)
                                    await page.goto(f"{BASE_URL}/s/{topic}?extensions[]=pdf")
                                    await page.wait_for_load_state("networkidle", timeout=20000)
                                    await page.wait_for_selector("z-bookcard.ready", timeout=15000)
                                    
                                    # Re-fetch bookcards after navigation
                                    bookcards = await page.locator("z-bookcard.ready").all()

                                except Exception as e:
                                    print(f"   ❌ Download failed: {e}")
                                    
                                    # --- CHECK IF WE HIT DAILY LIMIT ---
                                    try:
                                        page_content = await page.content()
                                        if "Daily limit reached" in page_content:
                                            print("\n🚫 DAILY LIMIT REACHED! Switching accounts...")
                                            await context.close()
                                            await browser.close()
                                            return "LIMIT_REACHED", download_count, downloaded_books
                                    except:
                                        pass
                                    
                                    # Try to go back to search on failure (keep PDF filter)
                                    try:
                                        await page.goto(f"{BASE_URL}/s/{topic}?extensions[]=pdf")
                                        await page.wait_for_load_state("networkidle", timeout=20000)
                                        await page.wait_for_selector("z-bookcard.ready", timeout=15000)
                                        bookcards = await page.locator("z-bookcard.ready").all()
                                    except:
                                        pass
                                
                            except Exception as e:
                                print(f"   ❌ Error processing book: {e}")
                
                except Exception as e:
                    print(f"⚠️ Session Error: {e}")
                finally:
                    # Always cleanup browser resources
                    try:
                        await context.close()
                    except:
                        pass
                    try:
                        await browser.close()
                    except:
                        pass
            
                return "Download & Indexing Complete.", download_count, downloaded_books

# --- 2. THE MCP TOOL (Wrapper - Callable by Claude) ---
@mcp.tool()
async def download_books_by_topic(topic: str, max_books: int = 9):
    """Downloads books, cleans metadata with LLM, and updates DB."""
    result, count, books = await core_download_logic(topic=topic, max_books=max_books)
    return f"{result} ({count} books downloaded)"

if __name__ == "__main__":
    mcp.run()