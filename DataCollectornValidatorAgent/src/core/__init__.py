# Core Package
# - agent: Main AI agent
# - memory: Supabase cloud memory
# - curated_papers: Topic definitions

from .memory import AgentMemory
from .curated_papers import CURATED_TOPICS, SURVEY_QUERIES, get_collection_plan
