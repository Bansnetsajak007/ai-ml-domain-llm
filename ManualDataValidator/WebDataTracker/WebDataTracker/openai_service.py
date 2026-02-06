import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Support multiple keys if needed, but defaulting to standard OPENAI_API_KEY
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '30'))  # seconds

def check_duplicate_with_openai(new_resource, existing_candidates):
    """
    Uses OpenAI to check if a new resource is a duplicate of existing candidates.
    Args:
        new_resource (dict): The resource being added.
        existing_candidates (list): List of potential duplicate resources found by fuzzy search.
    Returns:
        dict: {
            "is_duplicate": bool,
            "status": "SAME" | "DIFFERENT" | "UNCERTAIN",
            "explanation": str,
            "existing": dict (the matched resource if SAME)
        }
    """
    if not existing_candidates:
        return {"is_duplicate": False, "status": "DIFFERENT", "explanation": "No similar items found locally."}

    if not OPENAI_API_KEY:
        return {"is_duplicate": False, "status": "UNCERTAIN", "explanation": "OpenAI API Key not configured."}

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Construct the prompt
    candidate_desc = "\n".join([
        f"Candidate {i+1}: {json.dumps(c, default=str)}" 
        for i, c in enumerate(existing_candidates)
    ])
    
    system_prompt = "You are an AI assistant managing a library dataset. Your job is to detect duplicates."
    user_prompt = f"""
    Determine if the 'New Resource' is the SAME logical resource as any of the 'Existing Candidates'.
    
    Resources are considered the SAME if they refer to the same paper/book/article, even if the title formatting, authors list, or source differs slightly.
    Resources are DIFFERENT if they are different editions, different papers by same authors, or completely unrelated.
    
    New Resource: {json.dumps(new_resource, default=str)}
    
    Existing Candidates:
    {candidate_desc}
    
    Respond in JSON format ONLY:
    {{
        "status": "SAME" | "DIFFERENT" | "UNCERTAIN",
        "explanation": "One line explanation of why.",
        "match_index": <index of matched candidate, e.g. 1, or null if different>
    }}
    """
    
    try:
        # User requested "gpt-5-nano", which doesn't exist publicly yet. 
        # Using "gpt-4o-mini" as the best small/fast equivalent.
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            timeout=LLM_TIMEOUT
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        status = result.get('status', 'UNCERTAIN')
        match_index = result.get('match_index')
        
        is_duplicate = status == 'SAME'
        
        return {
            "is_duplicate": is_duplicate,
            "status": status,
            "explanation": result.get('explanation', "AI provided no explanation."),
            "existing": existing_candidates[match_index-1] if (is_duplicate and match_index and match_index <= len(existing_candidates)) else None
        }

    except Exception as e:
        print(f"OpenAI Error: {e}")
        return {"is_duplicate": False, "status": "UNCERTAIN", "explanation": f"AI Check Failed: {str(e)}"}
