import sqlite3

def init_db():
    conn = sqlite3.connect("shared_memory.db")
    cursor = conn.cursor()
    
    # We add 'authors', 'normalized_title', and 'full_json' columns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS downloads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            normalized_title TEXT,
            authors TEXT,
            filename TEXT,
            account_used TEXT,
            full_json TEXT,  -- We store the exact JSON you asked for here
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database upgraded with LLM metadata columns!")

if __name__ == "__main__":
    init_db()