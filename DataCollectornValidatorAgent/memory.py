"""
Memory System for AI Librarian Agent
=====================================
Stores embeddings of downloaded book metadata to prevent duplicates
across multiple users.

Flow:
1. BEFORE download: Read memory, check if similar book exists
2. AFTER download: Write new book metadata + embedding to memory
"""

import json
import os
import sqlite3
import numpy as np
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
client = OpenAI()

MEMORY_DB_PATH = "agent_memory.db"
SIMILARITY_THRESHOLD = 0.85  # Books with similarity > 85% are considered duplicates


def init_memory_db():
    """Initialize the memory database with embeddings support."""
    conn = sqlite3.connect(MEMORY_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS book_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            authors TEXT,
            source TEXT,
            search_topic TEXT,
            embedding BLOB,
            downloaded_by TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Index for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_normalized_title 
        ON book_memory(normalized_title)
    """)
    
    conn.commit()
    conn.close()
    print("✅ Memory database initialized!")


def get_embedding(text: str) -> list:
    """Generate embedding for text using OpenAI."""
    try:
        response = client.embeddings.create(
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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embedding_to_bytes(embedding: list) -> bytes:
    """Convert embedding list to bytes for storage."""
    return np.array(embedding, dtype=np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> list:
    """Convert bytes back to embedding list."""
    return np.frombuffer(data, dtype=np.float32).tolist()


class AgentMemory:
    """
    Memory system for the AI Librarian Agent.
    Stores and retrieves book information with semantic search.
    """
    
    def __init__(self):
        # Ensure DB exists
        if not os.path.exists(MEMORY_DB_PATH):
            init_memory_db()
        self._cache = {}  # Cache embeddings in memory for speed
    
    def _get_book_text(self, title: str, authors: str = "") -> str:
        """Create searchable text from book metadata."""
        return f"{title} by {authors}".strip()
    
    def check_duplicate(self, title: str, authors: str = "") -> dict:
        """
        Check if a similar book already exists in memory.
        
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
        
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, authors, embedding, downloaded_by 
            FROM book_memory 
            WHERE embedding IS NOT NULL
        """)
        
        max_similarity = 0.0
        most_similar = None
        
        for row in cursor.fetchall():
            book_id, stored_title, stored_authors, embedding_bytes, downloaded_by = row
            
            if embedding_bytes:
                stored_embedding = bytes_to_embedding(embedding_bytes)
                similarity = cosine_similarity(new_embedding, stored_embedding)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = {
                        "id": book_id,
                        "title": stored_title,
                        "authors": stored_authors,
                        "downloaded_by": downloaded_by,
                        "similarity": similarity
                    }
        
        conn.close()
        
        is_dup = max_similarity >= SIMILARITY_THRESHOLD
        
        return {
            "is_duplicate": is_dup,
            "similar_book": most_similar if is_dup else None,
            "similarity": max_similarity
        }
    
    def _check_exact_match(self, title: str) -> dict:
        """Fallback: Check for exact normalized title match."""
        normalized = title.lower().strip()
        
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, title, authors, downloaded_by 
            FROM book_memory 
            WHERE normalized_title = ?
        """, (normalized,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "is_duplicate": True,
                "similar_book": {
                    "id": row[0],
                    "title": row[1],
                    "authors": row[2],
                    "downloaded_by": row[3],
                    "similarity": 1.0
                },
                "similarity": 1.0
            }
        
        return {"is_duplicate": False, "similar_book": None, "similarity": 0.0}
    
    def add_book(self, title: str, authors: str, source: str, 
                 search_topic: str, downloaded_by: str) -> bool:
        """
        Add a new book to memory after downloading.
        
        Returns:
            True if added successfully, False otherwise
        """
        book_text = self._get_book_text(title, authors)
        embedding = get_embedding(book_text)
        
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO book_memory 
                (title, normalized_title, authors, source, search_topic, embedding, downloaded_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                title,
                title.lower().strip(),
                authors,
                source,
                search_topic,
                embedding_to_bytes(embedding) if embedding else None,
                downloaded_by
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to add book to memory: {e}")
            conn.close()
            return False
    
    def get_all_books(self) -> list:
        """Get all books in memory."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT title, authors, source, search_topic, downloaded_by, timestamp
            FROM book_memory
            ORDER BY timestamp DESC
        """)
        
        books = []
        for row in cursor.fetchall():
            books.append({
                "title": row[0],
                "authors": row[1],
                "source": row[2],
                "search_topic": row[3],
                "downloaded_by": row[4],
                "timestamp": row[5]
            })
        
        conn.close()
        return books
    
    def get_books_by_topic(self, topic: str) -> list:
        """Get all books downloaded for a specific topic."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT title, authors, downloaded_by
            FROM book_memory
            WHERE search_topic LIKE ?
        """, (f"%{topic}%",))
        
        books = [{"title": r[0], "authors": r[1], "downloaded_by": r[2]} 
                 for r in cursor.fetchall()]
        
        conn.close()
        return books
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM book_memory")
        total = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT downloaded_by, COUNT(*) 
            FROM book_memory 
            GROUP BY downloaded_by
        """)
        by_user = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT search_topic, COUNT(*) 
            FROM book_memory 
            GROUP BY search_topic
            ORDER BY COUNT(*) DESC
            LIMIT 10
        """)
        by_topic = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_books": total,
            "by_user": by_user,
            "top_topics": by_topic
        }
    
    def search_similar(self, query: str, limit: int = 5) -> list:
        """Search for books similar to a query."""
        query_embedding = get_embedding(query)
        
        if query_embedding is None:
            return []
        
        conn = sqlite3.connect(MEMORY_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT title, authors, embedding, downloaded_by 
            FROM book_memory 
            WHERE embedding IS NOT NULL
        """)
        
        results = []
        for row in cursor.fetchall():
            title, authors, embedding_bytes, downloaded_by = row
            
            if embedding_bytes:
                stored_embedding = bytes_to_embedding(embedding_bytes)
                similarity = cosine_similarity(query_embedding, stored_embedding)
                results.append({
                    "title": title,
                    "authors": authors,
                    "downloaded_by": downloaded_by,
                    "similarity": similarity
                })
        
        conn.close()
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]


# Initialize DB on import
if not os.path.exists(MEMORY_DB_PATH):
    init_memory_db()


# Quick test
if __name__ == "__main__":
    print("Testing Memory System...\n")
    
    memory = AgentMemory()
    
    # Add some test books
    print("Adding test books...")
    memory.add_book(
        title="Deep Learning",
        authors="Ian Goodfellow",
        source="Z-Library",
        search_topic="deep learning fundamentals",
        downloaded_by="User_1"
    )
    
    memory.add_book(
        title="Machine Learning: A Probabilistic Perspective",
        authors="Kevin Murphy",
        source="Z-Library", 
        search_topic="machine learning",
        downloaded_by="User_2"
    )
    
    # Check for duplicate
    print("\nChecking for duplicates...")
    result = memory.check_duplicate("Deep Learning by Goodfellow")
    print(f"  'Deep Learning by Goodfellow': {result}")
    
    result = memory.check_duplicate("Introduction to Deep Learning")
    print(f"  'Introduction to Deep Learning': {result}")
    
    # Get stats
    print("\nMemory Stats:")
    stats = memory.get_stats()
    print(f"  Total books: {stats['total_books']}")
    print(f"  By user: {stats['by_user']}")
