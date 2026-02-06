from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = 'ai_resource_tracker'

def get_db_connection():
    if not MONGO_URI:
        raise ValueError("No MONGO_URI found in environment variables")
    
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db

def init_db():
    """Initializes the database with necessary indexes."""
    db = get_db_connection()
    resources = db.resources
    
    # Text index for fuzzy matching (though we'll do logic in python mostly, this helps for search)
    resources.create_index([("normalized_title", "text")])
    
    # Unique indexes for strict duplicate prevention
    # Using sparse=True because unique_id might not always be present (e.g., blogs)
    resources.create_index("unique_id", unique=True, sparse=True)
    
    print("Database indexes initialized.")

if __name__ == "__main__":
    init_db()
