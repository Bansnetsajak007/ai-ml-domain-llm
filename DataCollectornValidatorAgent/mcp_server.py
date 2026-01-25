import asyncio
import json
import sqlite3
import os
import dotenv
from fastmcp import FastMCP
from playwright.async_api import async_playwright
from openai import OpenAI 

# --- CONFIGURATION ---
BASE_URL = "https://z-lib.sk"
DOWNLOAD_FOLDER = os.path.abspath("data/books")
DB_PATH = "shared_memory.db"
RESOURCES_FILE = "data/resources.json"

# Load environment variables
dotenv.load_dotenv()
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
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
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
    with open(RESOURCES_FILE, "r+") as f:
        data = json.load(f)
        data["resources"].append(metadata)
        f.seek(0)
        json.dump(data, f, indent=4)

def log_download_to_db(metadata, filename, account):
    """Updates SQLite with the Cleaned LLM Data"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO downloads 
        (title, normalized_title, authors, filename, account_used, full_json) 
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        metadata["title"], 
        metadata["normalized_title"], 
        json.dumps(metadata["authors"]), # Store list as string
        filename, 
        account,
        json.dumps(metadata) # Store the full object just in case
    ))
    conn.commit()
    conn.close()

def is_duplicate(normalized_title):
    """Checks DB using the CLEAN normalized title"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM downloads WHERE normalized_title = ?", (normalized_title,))
    res = cursor.fetchone()
    conn.close()
    return res is not None

# --- 1. THE CORE LOGIC (Plain Python - Callable by run.py) ---
async def core_download_logic(topic: str, accounts_to_use: int = 1, max_books: int = 9):
    """The actual Playwright automation logic. Returns (message, books_downloaded)."""
    
    total_downloaded_this_session = 0
    with open("accounts.json", "r") as f: accounts = json.load(f)

    async with async_playwright() as p:
        # headless=False so you can SEE it working
        browser = await p.chromium.launch(headless=False)
        
        for i in range(min(accounts_to_use, len(accounts))):
            current_acc = accounts[i]
            print(f"\n🚀 STARTING SESSION: {current_acc['name']}")

            context = await browser.new_context()
            await context.add_cookies([
                {'name': 'remix_userid', 'value': current_acc['remix_userid'], 'domain': '.z-lib.sk', 'path': '/'},
                {'name': 'remix_userkey', 'value': current_acc['remix_userkey'], 'domain': '.z-lib.sk', 'path': '/'}
            ])
            page = await context.new_page()
            
            # --- 🛑 MANUAL INTERVENTION BLOCK 🛑 ---
            try:
                print("Checking homepage...")
                await page.goto(f"{BASE_URL}/") 
                
                print("\n" + "="*50)
                print(f"🛑 MANUAL OVERRIDE: Please close the popup now!")
                print(f"⏳ Waiting 20 seconds for you to act...")
                print("="*50 + "\n")
                
                await asyncio.sleep(20) 
                print("✅ Resuming automation...")
                    
            except Exception as e:
                print(f"Navigation error: {e}")
            # ---------------------------------------
            
            try:
                print(f"🔎 Searching: {topic}...")
                await page.goto(f"{BASE_URL}/s/{topic}")
                
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
                    continue
                
                # Get all z-bookcard elements
                bookcards = await page.locator("z-bookcard.ready").all()
                
                if not bookcards:
                    print("⚠️ No book cards found on page.")
                    await page.screenshot(path="debug_no_books.png")
                    continue
                    
                print(f"📚 Found {len(bookcards)} books via z-bookcard elements...")
                # -------------------------------
                
                download_count = 0
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
                            
                            # UPDATE DB
                            log_download_to_db(clean_meta, proper_filename, current_acc['name'])
                            save_to_json_file(clean_meta)
                            
                            print(f"   ✅ Indexed: {clean_meta['title'][:50]}...")
                            download_count += 1
                            
                            print(f"   ⏳ Cooling down for 40s...")
                            await asyncio.sleep(40)
                            
                            # Go back to search results
                            await page.goto(f"{BASE_URL}/s/{topic}")
                            await page.wait_for_load_state("networkidle", timeout=20000)
                            await page.wait_for_selector("z-bookcard.ready", timeout=15000)
                            
                            # Re-fetch bookcards after navigation
                            bookcards = await page.locator("z-bookcard.ready").all()

                        except Exception as e:
                            print(f"   ❌ Download failed: {e}")
                            # Try to go back to search on failure
                            try:
                                await page.goto(f"{BASE_URL}/s/{topic}")
                                await page.wait_for_load_state("networkidle", timeout=20000)
                                await page.wait_for_selector("z-bookcard.ready", timeout=15000)
                                bookcards = await page.locator("z-bookcard.ready").all()
                            except:
                                pass
                        
                    except Exception as e:
                        print(f"   ❌ Error processing book: {e}")
                
                total_downloaded_this_session = download_count
            
            except Exception as e:
                print(f"Session Error: {e}")
            
            await context.close()

        await browser.close()
    return "Download & Indexing Complete.", total_downloaded_this_session

# --- 2. THE MCP TOOL (Wrapper - Callable by Claude) ---
@mcp.tool()
async def download_books_by_topic(topic: str, accounts_to_use: int = 1, max_books: int = 9):
    """Downloads books, cleans metadata with LLM, and updates DB."""
    result, count = await core_download_logic(topic, accounts_to_use, max_books)
    return f"{result} ({count} books downloaded)"

if __name__ == "__main__":
    mcp.run()