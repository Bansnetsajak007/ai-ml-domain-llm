"""
RAMESH Memory System - Supabase Edition ☁️
============================================
Cloud-based shared memory for multi-user duplicate detection.
Now supports BOTH Books (Z-Library) and Papers (arXiv)!

Why Supabase?
- 4 team members need to share the same memory
- SQLite is local-only, can't sync across PCs
- Supabase = Free PostgreSQL in the cloud
- Everyone connects to the same database!

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Sajak's PC ──────┐                                            │
│   Friend 1's PC ───┼──────►  SUPABASE CLOUD  ◄───── All users   │
│   Friend 2's PC ───┘         (shared memory)        see same    │
│                                                     data!       │
└─────────────────────────────────────────────────────────────────┘

Tables:
- book_memory: Books from Z-Library
- paper_memory: Research papers from arXiv

"""

import os
import numpy as np
from openai import OpenAI
from postgrest import SyncPostgrestClient
import dotenv

# Load config from config folder
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', '.env')
dotenv.load_dotenv(config_path)

# OpenAI for embeddings
openai_client = OpenAI()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Use the 'anon' public key

SIMILARITY_THRESHOLD = 0.85  # Books with similarity > 85% are considered duplicates

# Initialize PostgREST client
postgrest_client = None


def get_postgrest() -> SyncPostgrestClient:
    """Get or create PostgREST client for Supabase."""
    global postgrest_client
    if postgrest_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "❌ Supabase credentials not found!\n"
                "   Please add to your .env file:\n"
                "   SUPABASE_URL=your-project-url\n"
                "   SUPABASE_KEY=your-anon-key"
            )
        # Supabase REST API is at /rest/v1
        rest_url = f"{SUPABASE_URL}/rest/v1"
        postgrest_client = SyncPostgrestClient(
            rest_url,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}"
            }
        )
    return postgrest_client


def init_memory_db():
    """
    Initialize the Supabase table.
    Run this ONCE to create the table structure.
    """
    print("="*60)
    print("🚀 SUPABASE MEMORY SETUP")
    print("="*60)
    print("")
    print("To create the table, go to your Supabase Dashboard:")
    print("1. Open SQL Editor")
    print("2. Run this SQL:")
    print("")
    print("-"*60)
    sql = """
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

CREATE INDEX IF NOT EXISTS idx_normalized_title ON book_memory(normalized_title);
CREATE INDEX IF NOT EXISTS idx_downloaded_by ON book_memory(downloaded_by);
    """
    print(sql)
    print("-"*60)
    print("")
    print("3. Click 'Run'")
    print("")
    print("="*60)
    print("📄 PAPER MEMORY TABLE (for arXiv):")
    print("="*60)
    print("")
    paper_sql = """
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
    """
    print(paper_sql)
    print("-"*60)
    print("")
    print("✅ Then you're ready to use Ramesh with arXiv support!")
    print("="*60)
    print("")
    print("✅ Then you're ready to use Ramesh!")
    print("="*60)


def get_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",  # Fast and cheap
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Embedding generation failed: {e}")
        return None


def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class AgentMemory:
    """
    Cloud-based Memory System for RAMESH Agent.
    Uses Supabase for multi-user shared memory.
    
    🇳🇵 "Ramesh never forgets, and now your whole team won't either!"
    """
    
    def __init__(self):
        self.client = get_postgrest()
        self._cache = {}  # Local cache for speed
    
    def _get_book_text(self, title: str, authors: str = "") -> str:
        """Create searchable text from book metadata."""
        return f"{title} by {authors}".strip()
    
    def check_duplicate(self, title: str, authors: str = "") -> dict:
        """
        Check if a similar book already exists in shared memory.
        
        Returns:
            {
                "is_duplicate": bool,
                "similar_book": dict or None,
                "similarity": float
            }
        """
        book_text = self._get_book_text(title, authors)
        new_embedding = get_embedding(book_text)
        
        if new_embedding is None:
            # Fallback to exact title match if embedding fails
            return self._check_exact_match(title)
        
        # Fetch all embeddings from Supabase
        try:
            response = self.client.from_("book_memory").select(
                "id, title, authors, embedding, downloaded_by"
            ).not_.is_("embedding", "null").execute()
            
            max_similarity = 0.0
            most_similar = None
            
            for row in response.data:
                if row.get("embedding"):
                    stored_embedding = row["embedding"]
                    similarity = cosine_similarity(new_embedding, stored_embedding)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar = {
                            "id": row["id"],
                            "title": row["title"],
                            "authors": row["authors"],
                            "downloaded_by": row["downloaded_by"],
                            "similarity": similarity
                        }
            
            is_dup = max_similarity >= SIMILARITY_THRESHOLD
            
            return {
                "is_duplicate": is_dup,
                "similar_book": most_similar if is_dup else None,
                "similarity": max_similarity
            }
            
        except Exception as e:
            print(f"⚠️ Supabase query failed: {e}")
            return {"is_duplicate": False, "similar_book": None, "similarity": 0.0}
    
    def _check_exact_match(self, title: str) -> dict:
        """Fallback: Check for exact normalized title match."""
        normalized = title.lower().strip()
        
        try:
            response = self.client.from_("book_memory").select(
                "id, title, authors, downloaded_by"
            ).eq("normalized_title", normalized).execute()
            
            if response.data:
                row = response.data[0]
                return {
                    "is_duplicate": True,
                    "similar_book": {
                        "id": row["id"],
                        "title": row["title"],
                        "authors": row["authors"],
                        "downloaded_by": row["downloaded_by"],
                        "similarity": 1.0
                    },
                    "similarity": 1.0
                }
        except Exception as e:
            print(f"⚠️ Exact match check failed: {e}")
        
        return {"is_duplicate": False, "similar_book": None, "similarity": 0.0}
    
    def add_book(self, title: str, authors: str, source: str, 
                 search_topic: str, downloaded_by: str) -> bool:
        """
        Add a new book to shared cloud memory after downloading.
        
        Returns:
            True if added successfully, False otherwise
        """
        book_text = self._get_book_text(title, authors)
        embedding = get_embedding(book_text)
        
        try:
            data = {
                "title": title,
                "normalized_title": title.lower().strip(),
                "authors": authors,
                "source": source,
                "search_topic": search_topic,
                "embedding": embedding,
                "downloaded_by": downloaded_by
            }
            
            self.client.from_("book_memory").insert(data).execute()
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to add book to Supabase: {e}")
            return False
    
    def get_all_books(self) -> list:
        """Get all books in shared memory."""
        try:
            response = self.client.from_("book_memory").select(
                "title, authors, source, search_topic, downloaded_by, created_at"
            ).order("created_at", desc=True).execute()
            
            return [
                {
                    "title": row["title"],
                    "authors": row["authors"],
                    "source": row["source"],
                    "search_topic": row["search_topic"],
                    "downloaded_by": row["downloaded_by"],
                    "timestamp": row["created_at"]
                }
                for row in response.data
            ]
        except Exception as e:
            print(f"⚠️ Failed to fetch books: {e}")
            return []
    
    def get_books_by_topic(self, topic: str) -> list:
        """Get all books downloaded for a specific topic."""
        try:
            response = self.client.from_("book_memory").select(
                "title, authors, downloaded_by"
            ).ilike("search_topic", f"%{topic}%").execute()
            
            return [
                {"title": r["title"], "authors": r["authors"], "downloaded_by": r["downloaded_by"]} 
                for r in response.data
            ]
        except Exception as e:
            print(f"⚠️ Topic search failed: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Get memory statistics for the whole team."""
        try:
            # Get all books
            response = self.client.from_("book_memory").select(
                "downloaded_by, search_topic"
            ).execute()
            
            total = len(response.data)
            
            # Count by user
            by_user = {}
            by_topic = {}
            
            for row in response.data:
                user = row.get("downloaded_by", "unknown")
                topic = row.get("search_topic", "unknown")
                
                by_user[user] = by_user.get(user, 0) + 1
                by_topic[topic] = by_topic.get(topic, 0) + 1
            
            # Sort topics by count
            sorted_topics = dict(sorted(by_topic.items(), key=lambda x: x[1], reverse=True)[:10])
            
            return {
                "total_books": total,
                "by_user": by_user,
                "top_topics": sorted_topics
            }
            
        except Exception as e:
            print(f"⚠️ Stats query failed: {e}")
            return {"total_books": 0, "by_user": {}, "top_topics": {}}
    
    def search_similar(self, query: str, limit: int = 5) -> list:
        """Search for books similar to a query using embeddings."""
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            return []
        
        try:
            response = self.client.from_("book_memory").select(
                "title, authors, embedding, downloaded_by"
            ).not_.is_("embedding", "null").execute()
            
            results = []
            for row in response.data:
                if row.get("embedding"):
                    similarity = cosine_similarity(query_embedding, row["embedding"])
                    results.append({
                        "title": row["title"],
                        "authors": row["authors"],
                        "downloaded_by": row["downloaded_by"],
                        "similarity": similarity
                    })
            
            # Sort by similarity and return top results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"⚠️ Search failed: {e}")
            return []
    
    # ========================================
    # PAPER MEMORY METHODS (arXiv)
    # ========================================
    
    def check_paper_duplicate(self, arxiv_id: str, title: str) -> dict:
        """
        Check if a paper already exists in shared memory.
        First checks by arxiv_id (exact match), then by title similarity.
        
        Returns:
            {
                "is_duplicate": bool,
                "similar_paper": dict or None,
                "similarity": float
            }
        """
        # First, check exact arxiv_id match
        try:
            response = self.client.from_("paper_memory").select(
                "id, arxiv_id, title, authors, downloaded_by"
            ).eq("arxiv_id", arxiv_id).execute()
            
            if response.data:
                row = response.data[0]
                return {
                    "is_duplicate": True,
                    "similar_paper": {
                        "id": row["id"],
                        "arxiv_id": row["arxiv_id"],
                        "title": row["title"],
                        "authors": row["authors"],
                        "downloaded_by": row["downloaded_by"],
                        "similarity": 1.0
                    },
                    "similarity": 1.0
                }
        except Exception as e:
            print(f"⚠️ arXiv ID check failed: {e}")
        
        # If no exact match, check title similarity
        paper_text = f"{title}"
        new_embedding = get_embedding(paper_text)
        
        if new_embedding is None:
            return {"is_duplicate": False, "similar_paper": None, "similarity": 0.0}
        
        try:
            response = self.client.from_("paper_memory").select(
                "id, arxiv_id, title, authors, embedding, downloaded_by"
            ).not_.is_("embedding", "null").execute()
            
            max_similarity = 0.0
            most_similar = None
            
            for row in response.data:
                if row.get("embedding"):
                    stored_embedding = row["embedding"]
                    similarity = cosine_similarity(new_embedding, stored_embedding)
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar = {
                            "id": row["id"],
                            "arxiv_id": row["arxiv_id"],
                            "title": row["title"],
                            "authors": row["authors"],
                            "downloaded_by": row["downloaded_by"],
                            "similarity": similarity
                        }
            
            is_dup = max_similarity >= SIMILARITY_THRESHOLD
            
            return {
                "is_duplicate": is_dup,
                "similar_paper": most_similar if is_dup else None,
                "similarity": max_similarity
            }
            
        except Exception as e:
            print(f"⚠️ Paper similarity check failed: {e}")
            return {"is_duplicate": False, "similar_paper": None, "similarity": 0.0}
    
    def add_paper(self, arxiv_id: str, title: str, authors: str, 
                  abstract: str, categories: str, downloaded_by: str) -> bool:
        """
        Add a new paper to shared cloud memory after downloading.
        
        Returns:
            True if added successfully, False otherwise
        """
        paper_text = f"{title} by {authors}"
        embedding = get_embedding(paper_text)
        
        try:
            data = {
                "arxiv_id": arxiv_id,
                "title": title,
                "normalized_title": title.lower().strip(),
                "authors": authors,
                "abstract": abstract[:1000] if abstract else "",  # Limit abstract length
                "categories": categories,
                "embedding": embedding,
                "downloaded_by": downloaded_by
            }
            
            self.client.from_("paper_memory").insert(data).execute()
            return True
            
        except Exception as e:
            # Handle duplicate arxiv_id gracefully
            if "duplicate" in str(e).lower():
                print(f"   ⚠️ Paper already in memory: {arxiv_id}")
            else:
                print(f"⚠️ Failed to add paper to Supabase: {e}")
            return False
    
    def get_all_papers(self) -> list:
        """Get all papers in shared memory."""
        try:
            response = self.client.from_("paper_memory").select(
                "arxiv_id, title, authors, categories, downloaded_by, created_at"
            ).order("created_at", desc=True).execute()
            
            return [
                {
                    "arxiv_id": row["arxiv_id"],
                    "title": row["title"],
                    "authors": row["authors"],
                    "categories": row["categories"],
                    "downloaded_by": row["downloaded_by"],
                    "timestamp": row["created_at"]
                }
                for row in response.data
            ]
        except Exception as e:
            print(f"⚠️ Failed to fetch papers: {e}")
            return []
    
    def get_papers_by_category(self, category: str) -> list:
        """Get all papers in a specific arXiv category."""
        try:
            response = self.client.from_("paper_memory").select(
                "arxiv_id, title, authors, downloaded_by"
            ).ilike("categories", f"%{category}%").execute()
            
            return [
                {"arxiv_id": r["arxiv_id"], "title": r["title"], 
                 "authors": r["authors"], "downloaded_by": r["downloaded_by"]} 
                for r in response.data
            ]
        except Exception as e:
            print(f"⚠️ Category search failed: {e}")
            return []
    
    def get_paper_stats(self) -> dict:
        """Get memory statistics for papers."""
        try:
            response = self.client.from_("paper_memory").select(
                "downloaded_by, categories"
            ).execute()
            
            total = len(response.data)
            by_user = {}
            by_category = {}
            
            for row in response.data:
                user = row.get("downloaded_by", "unknown")
                categories = row.get("categories", "unknown")
                
                by_user[user] = by_user.get(user, 0) + 1
                
                # Parse categories (comma-separated)
                for cat in categories.split(","):
                    cat = cat.strip()
                    if cat:
                        by_category[cat] = by_category.get(cat, 0) + 1
            
            # Sort categories by count
            sorted_categories = dict(sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:10])
            
            return {
                "total_papers": total,
                "by_user": by_user,
                "top_categories": sorted_categories
            }
            
        except Exception as e:
            print(f"⚠️ Paper stats query failed: {e}")
            return {"total_papers": 0, "by_user": {}, "top_categories": {}}
    
    def search_papers_similar(self, query: str, limit: int = 5) -> list:
        """Search for papers similar to a query using embeddings."""
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            return []
        
        try:
            response = self.client.from_("paper_memory").select(
                "arxiv_id, title, authors, embedding, downloaded_by"
            ).not_.is_("embedding", "null").execute()
            
            results = []
            for row in response.data:
                if row.get("embedding"):
                    similarity = cosine_similarity(query_embedding, row["embedding"])
                    results.append({
                        "arxiv_id": row["arxiv_id"],
                        "title": row["title"],
                        "authors": row["authors"],
                        "downloaded_by": row["downloaded_by"],
                        "similarity": similarity
                    })
            
            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            print(f"⚠️ Paper search failed: {e}")
            return []
    
    def get_combined_stats(self) -> dict:
        """Get combined statistics for both books and papers."""
        book_stats = self.get_stats()
        paper_stats = self.get_paper_stats()
        
        return {
            "books": book_stats,
            "papers": paper_stats,
            "total_resources": book_stats.get("total_books", 0) + paper_stats.get("total_papers", 0)
        }


# Quick test / setup
if __name__ == "__main__":
    print("")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  RAMESH MEMORY SYSTEM - SUPABASE CLOUD EDITION ☁️                 ║")
    print("║  Now with Books (Z-Library) + Papers (arXiv) support!             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print("")
    
    # Check if credentials exist
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Supabase credentials not found!")
        print("")
        print("Please add these to your .env file:")
        print("   SUPABASE_URL=https://your-project.supabase.co")
        print("   SUPABASE_KEY=your-anon-public-key")
        print("")
        print("Get these from: Supabase Dashboard > Settings > API")
        print("")
        init_memory_db()  # Show SQL to create table
    else:
        print("✅ Supabase credentials found!")
        print(f"   URL: {SUPABASE_URL[:40]}...")
        print("")
        
        try:
            memory = AgentMemory()
            
            # Get combined stats
            combined = memory.get_combined_stats()
            book_stats = combined["books"]
            paper_stats = combined["papers"]
            
            print(f"✅ Connected to Supabase!")
            print(f"")
            print(f"📊 MEMORY STATISTICS")
            print(f"{'='*50}")
            print(f"   📚 Books in memory: {book_stats.get('total_books', 0)}")
            print(f"   📄 Papers in memory: {paper_stats.get('total_papers', 0)}")
            print(f"   📦 Total resources: {combined['total_resources']}")
            print(f"")
            
            if book_stats.get('by_user'):
                print(f"   👥 Book downloads by user:")
                for user, count in book_stats['by_user'].items():
                    print(f"      - {user}: {count} books")
            
            if paper_stats.get('by_user'):
                print(f"   👥 Paper downloads by user:")
                for user, count in paper_stats['by_user'].items():
                    print(f"      - {user}: {count} papers")
            
            if paper_stats.get('top_categories'):
                print(f"   🏷️ Top arXiv categories:")
                for cat, count in list(paper_stats['top_categories'].items())[:5]:
                    print(f"      - {cat}: {count} papers")
            
            print("")
            print("🙏 Ramesh is ready for the team!")
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("")
            init_memory_db()  # Show SQL to create table
