"""
RAMESH Curated Collection Runner 🚀
=====================================
Automated collection of 930 curated research papers.

This script runs the FULL collection automatically:
1. Collects papers topic by topic
2. Respects rate limits
3. Tracks progress
4. Avoids duplicates via memory
5. Generates detailed report

Usage:
    python scripts/run_curated_collection.py
    
Options:
    --dry-run       Show plan without downloading
    --resume        Resume from last checkpoint
    --topic TOPIC   Run only specific topic
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict
import dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load config from config folder
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', '.env')
dotenv.load_dotenv(config_path)

# Import our modules
from src.core.curated_papers import CURATED_TOPICS, SURVEY_QUERIES, get_collection_plan
from src.collectors.arxiv_collector import ArxivCollector, core_arxiv_download_logic
from src.core.memory import AgentMemory

# Configuration - use logs folder
LOGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
CHECKPOINT_FILE = os.path.join(LOGS_DIR, "collection_checkpoint.json")
REPORT_FILE = os.path.join(LOGS_DIR, "collection_report.json")
PAPERS_PER_QUERY_LIMIT = 15  # Max papers per individual query to avoid redundancy


class CuratedCollectionRunner:
    """
    Automated runner for curated paper collection.
    """
    
    def __init__(self, user_name: str = "sajak"):
        self.user_name = user_name
        self.checkpoint = self._load_checkpoint()
        self.stats = {
            "started_at": datetime.now().isoformat(),
            "total_downloaded": 0,
            "by_category": {},
            "by_topic": {},
            "failed_queries": [],
            "skipped_duplicates": 0
        }
        
        # Initialize memory
        try:
            self.memory = AgentMemory()
            print("✅ Connected to shared memory")
        except Exception as e:
            print(f"⚠️ Memory connection failed: {e}")
            self.memory = None
    
    def _load_checkpoint(self) -> dict:
        """Load checkpoint for resume capability."""
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, "r") as f:
                    return json.load(f)
            except:
                pass
        return {"completed_topics": [], "completed_surveys": []}
    
    def _save_checkpoint(self):
        """Save checkpoint for resume."""
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def _save_report(self):
        """Save collection report."""
        self.stats["completed_at"] = datetime.now().isoformat()
        with open(REPORT_FILE, "w") as f:
            json.dump(self.stats, f, indent=2)
    
    async def collect_topic(self, category: str, topic_name: str, config: dict) -> int:
        """Collect papers for a single topic."""
        topic_key = f"{category}:{topic_name}"
        
        # Skip if already completed
        if topic_key in self.checkpoint["completed_topics"]:
            print(f"   ⏭️ Already completed, skipping...")
            return 0
        
        total_collected = 0
        queries = config["queries"]
        categories = config.get("categories", ["cs.LG"])
        papers_per_query = min(config.get("papers_per_query", 10), PAPERS_PER_QUERY_LIMIT)
        
        print(f"\n   📝 Running {len(queries)} queries, {papers_per_query} papers each...")
        
        async with ArxivCollector(memory=self.memory, user_name=self.user_name) as collector:
            for query in queries:
                print(f"\n   🔎 Query: '{query}'")
                
                try:
                    message, count, papers = await collector.collect_papers(
                        query=query,
                        max_papers=papers_per_query,
                        categories=categories
                    )
                    
                    total_collected += count
                    print(f"   ✅ Downloaded: {count} papers")
                    
                except Exception as e:
                    print(f"   ❌ Query failed: {e}")
                    self.stats["failed_queries"].append({
                        "topic": topic_name,
                        "query": query,
                        "error": str(e)
                    })
                
                # Small delay between queries
                await asyncio.sleep(2)
        
        # Mark as completed
        self.checkpoint["completed_topics"].append(topic_key)
        self._save_checkpoint()
        
        return total_collected
    
    async def collect_surveys(self) -> int:
        """Collect survey/review papers."""
        total_collected = 0
        
        print("\n" + "=" * 60)
        print("📖 COLLECTING SURVEY PAPERS")
        print("=" * 60)
        
        async with ArxivCollector(memory=self.memory, user_name=self.user_name) as collector:
            for i, survey in enumerate(SURVEY_QUERIES):
                survey_key = survey["query"]
                
                if survey_key in self.checkpoint["completed_surveys"]:
                    print(f"\n   ⏭️ Survey '{survey_key[:40]}...' already completed")
                    continue
                
                print(f"\n   [{i+1}/{len(SURVEY_QUERIES)}] Survey: '{survey['query']}'")
                
                try:
                    message, count, papers = await collector.collect_papers(
                        query=survey["query"],
                        max_papers=survey["max"],
                        categories=survey.get("categories")
                    )
                    
                    total_collected += count
                    print(f"   ✅ Downloaded: {count} survey papers")
                    
                    self.checkpoint["completed_surveys"].append(survey_key)
                    self._save_checkpoint()
                    
                except Exception as e:
                    print(f"   ❌ Survey failed: {e}")
                    self.stats["failed_queries"].append({
                        "type": "survey",
                        "query": survey["query"],
                        "error": str(e)
                    })
                
                await asyncio.sleep(2)
        
        return total_collected
    
    async def run_full_collection(self, dry_run: bool = False):
        """Run the complete collection."""
        plan = get_collection_plan()
        
        print("\n" + "=" * 70)
        print("🚀 RAMESH CURATED COLLECTION - STARTING")
        print("=" * 70)
        print(f"👤 User: {self.user_name}")
        print(f"📊 Target: {plan['total_papers']} papers")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if dry_run:
            print("\n🔍 DRY RUN MODE - No downloads")
            from curated_papers import print_collection_plan
            print_collection_plan()
            return
        
        # Collect by category and topic
        for category_name, topics in CURATED_TOPICS.items():
            print("\n" + "=" * 60)
            print(f"🏷️  CATEGORY: {category_name.upper().replace('_', ' ')}")
            print("=" * 60)
            
            category_total = 0
            
            for topic_name, config in topics.items():
                print(f"\n📌 TOPIC: {topic_name}")
                print(f"   Target: {config['total_target']} papers")
                
                count = await self.collect_topic(category_name, topic_name, config)
                category_total += count
                
                self.stats["by_topic"][topic_name] = count
            
            self.stats["by_category"][category_name] = category_total
            self.stats["total_downloaded"] += category_total
            
            print(f"\n✅ Category '{category_name}' complete: {category_total} papers")
        
        # Collect surveys
        survey_count = await self.collect_surveys()
        self.stats["by_category"]["surveys"] = survey_count
        self.stats["total_downloaded"] += survey_count
        
        # Final report
        self._save_report()
        
        print("\n" + "=" * 70)
        print("🎉 COLLECTION COMPLETE!")
        print("=" * 70)
        print(f"📊 Total papers downloaded: {self.stats['total_downloaded']}")
        print(f"📁 Saved to: data/papers/")
        print(f"📋 Report: {REPORT_FILE}")
        
        if self.stats["failed_queries"]:
            print(f"⚠️ Failed queries: {len(self.stats['failed_queries'])}")
        
        print("=" * 70)
        
        return self.stats


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAMESH Curated Collection Runner")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--user", default="sajak", help="User name for tracking")
    
    args = parser.parse_args()
    
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ██████╗  █████╗ ███╗   ███╗███████╗███████╗██╗  ██╗                         ║
║   ██╔══██╗██╔══██╗████╗ ████║██╔════╝██╔════╝██║  ██║                         ║
║   ██████╔╝███████║██╔████╔██║█████╗  ███████╗███████║                         ║
║   ██╔══██╗██╔══██║██║╚██╔╝██║██╔══╝  ╚════██║██╔══██║                         ║
║   ██║  ██║██║  ██║██║ ╚═╝ ██║███████╗███████║██║  ██║                         ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝                         ║
║                                                                               ║
║              🚀 CURATED COLLECTION MODE - 930 Quality Papers                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if not args.resume and os.path.exists(CHECKPOINT_FILE):
        print("⚠️ Previous checkpoint found. Use --resume to continue, or delete checkpoint file.")
        response = input("   Start fresh? (y/n): ").strip().lower()
        if response == 'y':
            os.remove(CHECKPOINT_FILE)
            print("   ✅ Checkpoint cleared")
        else:
            print("   Using --resume mode")
            args.resume = True
    
    runner = CuratedCollectionRunner(user_name=args.user)
    
    try:
        await runner.run_full_collection(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\n⏸️ Collection paused. Use --resume to continue later.")
        runner._save_checkpoint()
        runner._save_report()


if __name__ == "__main__":
    asyncio.run(main())
