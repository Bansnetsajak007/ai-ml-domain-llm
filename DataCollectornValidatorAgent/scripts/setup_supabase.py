"""
RAMESH Database Setup Script
=============================
Run this script to set up the required Supabase tables.

Usage:
    python setup_supabase.py

This will display the SQL commands needed to create the tables.
You'll need to run these in your Supabase SQL Editor.
"""

def print_setup_sql():
    """Print the SQL needed to set up Supabase tables."""
    
    print("")
    print("=" * 70)
    print("🚀 RAMESH SUPABASE DATABASE SETUP")
    print("=" * 70)
    print("")
    print("Copy and run the following SQL in your Supabase SQL Editor:")
    print("(Supabase Dashboard > SQL Editor > New Query)")
    print("")
    print("-" * 70)
    print("")
    
    sql = """
-- =============================================
-- RAMESH MEMORY TABLES
-- =============================================

-- Table 1: Book Memory (Z-Library)
CREATE TABLE IF NOT EXISTS book_memory (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    authors TEXT,
    source TEXT,
    search_topic TEXT,
    embedding FLOAT8[],
    downloaded_by TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_book_normalized_title ON book_memory(normalized_title);
CREATE INDEX IF NOT EXISTS idx_book_downloaded_by ON book_memory(downloaded_by);

-- Table 2: Paper Memory (arXiv) - NEW!
CREATE TABLE IF NOT EXISTS paper_memory (
    id SERIAL PRIMARY KEY,
    arxiv_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    categories TEXT,
    embedding FLOAT8[],
    downloaded_by TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_paper_arxiv_id ON paper_memory(arxiv_id);
CREATE INDEX IF NOT EXISTS idx_paper_normalized_title ON paper_memory(normalized_title);
CREATE INDEX IF NOT EXISTS idx_paper_downloaded_by ON paper_memory(downloaded_by);
CREATE INDEX IF NOT EXISTS idx_paper_categories ON paper_memory(categories);

-- =============================================
-- Row Level Security (Optional but recommended)
-- =============================================

-- Enable RLS
ALTER TABLE book_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_memory ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated and anon users
-- (Since we're using the anon key for team access)
CREATE POLICY "Allow all for anon" ON book_memory FOR ALL USING (true);
CREATE POLICY "Allow all for anon" ON paper_memory FOR ALL USING (true);

-- =============================================
-- Done!
-- =============================================
"""
    
    print(sql)
    print("-" * 70)
    print("")
    print("✅ After running the SQL, RAMESH will be ready to use!")
    print("")
    print("📋 Quick verification query:")
    print("   SELECT COUNT(*) FROM book_memory;")
    print("   SELECT COUNT(*) FROM paper_memory;")
    print("")
    print("=" * 70)


def verify_connection():
    """Verify Supabase connection works."""
    import os
    import dotenv
    
    dotenv.load_dotenv()
    
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        print("❌ Supabase credentials not found in .env")
        print("   Please add SUPABASE_URL and SUPABASE_KEY")
        return False
    
    try:
        from postgrest import SyncPostgrestClient
        
        rest_url = f"{url}/rest/v1"
        client = SyncPostgrestClient(
            rest_url,
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}"
            }
        )
        
        # Try to query book_memory
        try:
            response = client.from_("book_memory").select("id").limit(1).execute()
            print(f"✅ book_memory table exists! ({len(response.data)} records checked)")
        except Exception as e:
            if "does not exist" in str(e):
                print("⚠️ book_memory table doesn't exist yet. Run the SQL above!")
            else:
                print(f"⚠️ book_memory query error: {e}")
        
        # Try to query paper_memory
        try:
            response = client.from_("paper_memory").select("id").limit(1).execute()
            print(f"✅ paper_memory table exists! ({len(response.data)} records checked)")
        except Exception as e:
            if "does not exist" in str(e):
                print("⚠️ paper_memory table doesn't exist yet. Run the SQL above!")
            else:
                print(f"⚠️ paper_memory query error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    print_setup_sql()
    
    print("\n🔍 Checking Supabase connection...")
    verify_connection()
