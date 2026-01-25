import asyncio
# CHANGE THIS LINE: Import the new logic function
from mcp_server import core_download_logic

TOPICS_TO_DOWNLOAD = [
    "Machine Learning Mathematics", 
    "Statistics for Data Science", 
    "Linear Algebra for AI"
]

ACCOUNT_TO_USE = 1
MAX_DOWNLOADS_PER_DAY = 9  # Global limit to avoid hitting Z-Library's daily cap

async def main():
    print("🤖 MANUALLY STARTING AGENT...")
    print(f"📊 Daily limit set to: {MAX_DOWNLOADS_PER_DAY} books\n")
    
    total_downloaded = 0
    
    for topic in TOPICS_TO_DOWNLOAD:
        if total_downloaded >= MAX_DOWNLOADS_PER_DAY:
            print(f"\n🛑 Daily limit reached ({MAX_DOWNLOADS_PER_DAY} books). Stopping.")
            break
            
        remaining = MAX_DOWNLOADS_PER_DAY - total_downloaded
        print(f"\n--- 🎯 TARGETING TOPIC: {topic} ---")
        print(f"📥 Can download up to {remaining} more books")
        
        # Pass the remaining limit to the download function
        result, books_downloaded = await core_download_logic(topic, ACCOUNT_TO_USE, max_books=remaining)
        
        total_downloaded += books_downloaded
        print(f"REPORT: {result}")
        print(f"📊 Total downloaded so far: {total_downloaded}/{MAX_DOWNLOADS_PER_DAY}")
        print("------------------------------------------------")
    
    print(f"\n✅ SESSION COMPLETE: Downloaded {total_downloaded} books total.")

if __name__ == "__main__":
    asyncio.run(main())