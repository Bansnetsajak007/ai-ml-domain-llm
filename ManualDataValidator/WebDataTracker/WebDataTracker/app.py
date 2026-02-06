from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timezone
from database import get_db_connection
from openai_service import check_duplicate_with_openai
from difflib import SequenceMatcher
from pymongo.errors import DuplicateKeyError

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret')

# Helper to normalize titles
def normalize_string(s):
    if not s:
        return ""
    return "".join(e for e in s.lower() if e.isalnum())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add')
def add_resource_page():
    return render_template('add_resource.html')

@app.route('/api/resources', methods=['GET'])
def get_resources():
    try:
        db = get_db_connection()
        # Get latest 50 resources
        resources = list(db.resources.find({}, {'_id': 0}).sort('added_at', -1).limit(50))
        return jsonify(resources)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/resources', methods=['POST'])
def add_resource():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        collector_name = data.get('collector_name')
        # The textarea might contain the JSON directly or wrapped
        # The user's example shows { "resources": [...] }
        raw_resource_data = data.get('resource_data') 
        force_save = data.get('force_save', False)

        if not collector_name or not raw_resource_data:
            return jsonify({"error": "Missing collector name or resource data"}), 400
            
        # Parse resource data (it's passed as a dict/list from the frontend JSON parse)
        # If the user pasted the structure { "resources": [...] }, raw_resource_data is that dict.
        
        resource_list = []
        if isinstance(raw_resource_data, dict) and 'resources' in raw_resource_data:
            resource_list = raw_resource_data['resources']
        elif isinstance(raw_resource_data, list):
            resource_list = raw_resource_data
        elif isinstance(raw_resource_data, dict):
            resource_list = [raw_resource_data]
        else:
            return jsonify({"error": "Invalid JSON format. Expected object with 'resources' array."}), 400

        db = get_db_connection()
        saved_count = 0
        skipped_count = 0
        
        for index, item in enumerate(resource_list):
            # Field Mapping
            # User format: resource_type -> type, arxiv_id -> unique_id
            
            # Helper to safely get unique ID
            unique_id = item.get('unique_id') or item.get('arxiv_id') or item.get('isbn')
            
            # Handle Title logic
            title = item.get('title')
            
            # If title is missing but we have a unique_id (like arxiv_id), use that as the title
            if not title and unique_id:
                title = f"arXiv:{unique_id}" if item.get('source') == 'arXiv' else unique_id
            
            if not title:
                # If still no title (and no unique_id), this is an invalid entry (like the empty ones at end of user json)
                # We log/skip or return error. User said "modify code to work only with this", 
                # and "this" has empty objects { "resource_type": "ResearchPaper", "source": "arXiv" }.
                # We should probably just skip them or warn. Let's return error for specific index to be helpful, 
                # OR just skip if it looks completely empty context. 
                # Let's skip if it lacks both title and ID.
                print(f"Skipping index {index}: No title or unique_id.")
                continue
                
            normalized_title = item.get('normalized_title') 
            if not normalized_title or normalized_title == "<normalized form of title >":
                normalized_title = normalize_string(title)
            
            resource_doc = {
                "type": item.get('resource_type', item.get('type', 'unknown')),
                "title": title,
                "normalized_title": normalized_title,
                "authors": item.get('authors', []),
                "year": item.get('year'),
                "source": item.get('source'),
                "unique_id": unique_id,
                "added_by": collector_name,
                "added_at": datetime.now(timezone.utc).isoformat()
            }

            # 1. Deterministic Check (Layer 1)
            # Check Unique ID
            if resource_doc.get('unique_id'):
                exact_match = db.resources.find_one({"unique_id": resource_doc['unique_id']}, {'_id': 0})
                if exact_match:
                    # Idempotency: If exact match exists, we SKIP it (assuming it's already saved).
                    skipped_count += 1
                    continue

            # Check Normalized Title (Exact)
            title_match = db.resources.find_one({"normalized_title": normalized_title}, {'_id': 0})
            if title_match:
                # Same logic: If exactly in DB, treat as success/skip.
                skipped_count += 1
                continue

            # 2. Fuzzy Matching (Layer 2)
            if not force_save:
                # We only check fuzzy if we aren't force-saving.
                # If we are force-saving, we skip this and insert.
                
                # Optimization: only check against items with same type? Nah, title similarity is cleaner.
                all_titles = list(db.resources.find({}, {"normalized_title": 1, "title": 1, "authors": 1, "_id": 0}))
                
                suspects = []
                for db_item in all_titles:
                    ratio = SequenceMatcher(None, normalized_title, db_item.get('normalized_title', '')).ratio()
                    # Lowered threshold to 0.6 to catch more potential variations (e.g. "Intro to..." vs "Deep Learning")
                    # We rely on Gemini (Layer 3) to dismiss false positives, so better to be over-sensitive here.
                    if ratio > 0.6: 
                        suspects.append(db_item)
                
                if suspects:
                    # Layer 3: OpenAI Check
                    best_candidate = suspects[0]
                    ai_result = check_duplicate_with_openai(resource_doc, [best_candidate])
                    
                    if ai_result['status'] in ['SAME', 'UNCERTAIN']:
                        # We STOP the batch here and return the warning.
                        return jsonify({
                            "error": "Potential Duplicate", 
                            "explanation": ai_result['explanation'],
                            "existing": best_candidate,
                            "processed_count": saved_count,
                            "problem_index": index
                        }), 409

            # Save
            try:
                db.resources.insert_one(resource_doc)
                saved_count += 1
            except DuplicateKeyError as e:
                # Handle unique index violation
                print(f"Duplicate key error at index {index}: {e}")
                skipped_count += 1
                continue
            
        return jsonify({
            "message": f"Batch processed. Saved: {saved_count}, Skipped (Already Existed): {skipped_count}",
            "saved_count": saved_count,
            "skipped_count": skipped_count
        }), 201

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/api/resources', methods=['DELETE'])
def delete_resources():
    try:
        # Check query params
        delete_all = request.args.get('all')
        collector = request.args.get('collector')
        
        db = get_db_connection()
        
        if delete_all == 'true':
            result = db.resources.delete_many({})
            return jsonify({"message": f"Deleted all {result.deleted_count} resources."}), 200
            
        if collector:
            result = db.resources.delete_many({"added_by": collector})
            return jsonify({"message": f"Deleted {result.deleted_count} resources added by '{collector}'."}), 200
            
        return jsonify({"error": "No valid delete criteria provided (use ?all=true or ?collector=Name)"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
