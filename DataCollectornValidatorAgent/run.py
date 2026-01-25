import asyncio
import json
from mcp_server import core_download_logic

TOPICS_TO_DOWNLOAD = [
    "Machine Learning Mathematics", 
    "Statistics for Data Science", 
    "Linear Algebra for AI"
]

DOWNLOADS_PER_ACCOUNT = 9  # Limit per account to avoid hitting Z-Library's daily cap

async def main():
    # Load accounts
    with open("accounts.json", "r") as f:
        accounts = json.load(f)
    
    total_accounts = len(accounts)
    max_possible = total_accounts * DOWNLOADS_PER_ACCOUNT
    
    print("🤖 MANUALLY STARTING AGENT...")
    print(f"📊 Accounts available: {total_accounts}")
    print(f"📊 Downloads per account: {DOWNLOADS_PER_ACCOUNT}")
    print(f"📊 Maximum possible downloads: {max_possible}\n")
    
    # Track state across all accounts
    current_account_idx = 0
    downloads_on_current_account = 0
    total_downloaded = 0
    topic_idx = 0
    
    while topic_idx < len(TOPICS_TO_DOWNLOAD) and current_account_idx < total_accounts:
        topic = TOPICS_TO_DOWNLOAD[topic_idx]
        current_account = accounts[current_account_idx]
        
        # Calculate how many more we can download on this account
        remaining_on_account = DOWNLOADS_PER_ACCOUNT - downloads_on_current_account
        
        if remaining_on_account <= 0:
            # Switch to next account
            current_account_idx += 1
            downloads_on_current_account = 0
            print(f"\n🔄 ROTATING TO NEXT ACCOUNT...")
            
            if current_account_idx >= total_accounts:
                print("🛑 All accounts exhausted!")
                break
            continue
        
        print(f"\n{'='*60}")
        print(f"🎯 TOPIC: {topic}")
        print(f"👤 Using: {current_account['name']} ({remaining_on_account} downloads remaining)")
        print(f"{'='*60}")
        
        # Download books for this topic with this account
        result, books_downloaded = await core_download_logic(
            topic=topic,
            account=current_account,
            max_books=remaining_on_account
        )
        
        downloads_on_current_account += books_downloaded
        total_downloaded += books_downloaded
        
        print(f"\n📊 {current_account['name']}: {downloads_on_current_account}/{DOWNLOADS_PER_ACCOUNT} used")
        print(f"📊 Total downloaded: {total_downloaded}")
        print(f"REPORT: {result}")
        
        # Move to next topic
        topic_idx += 1
        
        # If account is exhausted but more topics remain, rotate
        if downloads_on_current_account >= DOWNLOADS_PER_ACCOUNT and topic_idx < len(TOPICS_TO_DOWNLOAD):
            current_account_idx += 1
            downloads_on_current_account = 0
            
            if current_account_idx < total_accounts:
                print(f"\n🔄 Account limit reached. Switching to {accounts[current_account_idx]['name']}...")
            else:
                print("\n🛑 All accounts exhausted!")
                break
    
    print(f"\n{'='*60}")
    print(f"✅ SESSION COMPLETE")
    print(f"📚 Total books downloaded: {total_downloaded}")
    print(f"👤 Accounts used: {current_account_idx + 1}/{total_accounts}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())