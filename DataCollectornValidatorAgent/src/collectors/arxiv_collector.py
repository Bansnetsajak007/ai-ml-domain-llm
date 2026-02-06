"""
RAMESH arXiv Collector Module 📚🔬
====================================
Production-grade research paper collector from arXiv.

Features:
- Uses arXiv API (no authentication needed!)
- Bulk download capabilities (4k-5k papers)
- Rate limiting to respect arXiv's guidelines
- Automatic PDF download with metadata extraction
- Duplicate detection via shared memory
- Resumable downloads with progress tracking

arXiv API Guidelines:
- Rate limit: 1 request per 3 seconds
- Be polite: use bulk queries instead of many small ones
- Max 2000 results per query (pagination required)
"""

import asyncio
import aiohttp
import aiofiles
import os
import re
import json
import ssl
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import urllib.parse
import dotenv

# Try to use certifi for SSL certificates (fixes Windows SSL issues)
try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    # Fallback: create context that doesn't verify (less secure but works)
    SSL_CONTEXT = ssl.create_default_context()
    SSL_CONTEXT.check_hostname = False
    SSL_CONTEXT.verify_mode = ssl.CERT_NONE
    print("⚠️ certifi not found, using unverified SSL (install certifi for secure connections)")

# Load environment variables from config folder
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', '.env')
dotenv.load_dotenv(config_path)

# --- CONFIGURATION ---
ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DOWNLOAD_FOLDER = os.path.abspath(os.getenv("ARXIV_DATA_FOLDER", os.path.join(DATA_DIR, "papers")))
RESOURCES_FILE = os.getenv("ARXIV_RESOURCES_FILE", os.path.join(DATA_DIR, "arxiv_resources.json"))

# Rate limiting (arXiv guidelines: 1 request per 3 seconds)
API_RATE_LIMIT = float(os.getenv("ARXIV_API_RATE_LIMIT", "3.0"))  
PDF_DOWNLOAD_DELAY = float(os.getenv("ARXIV_PDF_DELAY", "1.0"))
MAX_RESULTS_PER_QUERY = int(os.getenv("ARXIV_MAX_PER_QUERY", "100"))
MAX_RETRIES = int(os.getenv("ARXIV_MAX_RETRIES", "3"))
DOWNLOAD_TIMEOUT = int(os.getenv("ARXIV_DOWNLOAD_TIMEOUT", "120"))

# Ensure folders exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(RESOURCES_FILE) if os.path.dirname(RESOURCES_FILE) else ".", exist_ok=True)
if not os.path.exists(RESOURCES_FILE):
    with open(RESOURCES_FILE, "w") as f:
        json.dump({"papers": []}, f)


class ArxivCategory(Enum):
    """Common arXiv categories for CS/AI/ML research."""
    CS_AI = "cs.AI"           # Artificial Intelligence
    CS_CL = "cs.CL"           # Computation and Language (NLP)
    CS_CV = "cs.CV"           # Computer Vision
    CS_LG = "cs.LG"           # Machine Learning
    CS_NE = "cs.NE"           # Neural and Evolutionary Computing
    CS_IR = "cs.IR"           # Information Retrieval
    STAT_ML = "stat.ML"       # Statistics - Machine Learning
    MATH_OC = "math.OC"       # Optimization and Control
    CS_RO = "cs.RO"           # Robotics
    CS_DC = "cs.DC"           # Distributed Computing
    CS_CR = "cs.CR"           # Cryptography and Security
    CS_DB = "cs.DB"           # Databases
    CS_SE = "cs.SE"           # Software Engineering
    CS_PL = "cs.PL"           # Programming Languages
    EESS_SP = "eess.SP"       # Signal Processing
    QUANT_PH = "quant-ph"     # Quantum Physics


@dataclass
class ArxivPaper:
    """Represents an arXiv paper with metadata."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    primary_category: str
    published: str
    updated: str
    pdf_url: str
    abs_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    
    @property
    def clean_filename(self) -> str:
        """Generate a clean filename for the PDF."""
        # Remove special characters, limit length
        clean_title = re.sub(r'[^\w\s-]', '', self.title)
        clean_title = re.sub(r'\s+', '_', clean_title.strip())
        clean_title = clean_title[:100]  # Limit length
        arxiv_id_clean = self.arxiv_id.replace("/", "_").replace(".", "_")
        return f"{arxiv_id_clean}_{clean_title}.pdf"
    
    @property
    def authors_str(self) -> str:
        """Get authors as comma-separated string."""
        return ", ".join(self.authors[:5])  # Limit to first 5 authors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)


class ArxivAPIError(Exception):
    """Custom exception for arXiv API errors."""
    pass


class ArxivCollector:
    """
    Production-grade arXiv paper collector.
    
    Features:
    - Async HTTP for efficient bulk downloads
    - Rate limiting for API compliance
    - Automatic retry with exponential backoff
    - Progress tracking and resumability
    - Memory integration for deduplication
    """
    
    def __init__(self, memory=None, user_name: str = "unknown"):
        self.memory = memory
        self.user_name = user_name
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_api_call = 0.0
        self.downloaded_papers: List[Dict] = []
        self.stats = {
            "searched": 0,
            "found": 0,
            "downloaded": 0,
            "skipped_duplicate": 0,
            "failed": 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
        # Use custom SSL context for Windows compatibility
        connector = aiohttp.TCPConnector(ssl=SSL_CONTEXT)
        self.session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Enforce rate limiting for arXiv API."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_api_call
        if elapsed < API_RATE_LIMIT:
            await asyncio.sleep(API_RATE_LIMIT - elapsed)
        self.last_api_call = asyncio.get_event_loop().time()
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ArxivPaper]:
        """Parse arXiv API XML response into ArxivPaper objects."""
        papers = []
        
        # Define namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        try:
            root = ET.fromstring(xml_content)
            
            for entry in root.findall('atom:entry', namespaces):
                # Extract basic metadata
                arxiv_id_element = entry.find('atom:id', namespaces)
                if arxiv_id_element is None:
                    continue
                    
                arxiv_id_raw = arxiv_id_element.text
                # Extract ID from URL (e.g., http://arxiv.org/abs/2301.00001v1 -> 2301.00001)
                arxiv_id = arxiv_id_raw.split('/abs/')[-1]
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)  # Remove version
                
                title_el = entry.find('atom:title', namespaces)
                title = title_el.text.strip().replace('\n', ' ') if title_el is not None else "Unknown"
                
                # Get abstract
                summary_el = entry.find('atom:summary', namespaces)
                abstract = summary_el.text.strip().replace('\n', ' ') if summary_el is not None else ""
                
                # Get authors
                authors = []
                for author in entry.findall('atom:author', namespaces):
                    name_el = author.find('atom:name', namespaces)
                    if name_el is not None:
                        authors.append(name_el.text)
                
                # Get categories
                categories = []
                primary_category = ""
                for category in entry.findall('atom:category', namespaces):
                    term = category.get('term', '')
                    if term:
                        categories.append(term)
                
                # Primary category from arxiv namespace
                primary_cat_el = entry.find('arxiv:primary_category', namespaces)
                if primary_cat_el is not None:
                    primary_category = primary_cat_el.get('term', categories[0] if categories else '')
                elif categories:
                    primary_category = categories[0]
                
                # Get dates
                published_el = entry.find('atom:published', namespaces)
                published = published_el.text if published_el is not None else ""
                
                updated_el = entry.find('atom:updated', namespaces)
                updated = updated_el.text if updated_el is not None else ""
                
                # Get URLs
                pdf_url = f"{ARXIV_PDF_BASE}/{arxiv_id}.pdf"
                abs_url = f"https://arxiv.org/abs/{arxiv_id}"
                
                # Optional fields
                comment_el = entry.find('arxiv:comment', namespaces)
                comment = comment_el.text if comment_el is not None else None
                
                journal_el = entry.find('arxiv:journal_ref', namespaces)
                journal_ref = journal_el.text if journal_el is not None else None
                
                doi_el = entry.find('arxiv:doi', namespaces)
                doi = doi_el.text if doi_el is not None else None
                
                paper = ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    categories=categories,
                    primary_category=primary_category,
                    published=published,
                    updated=updated,
                    pdf_url=pdf_url,
                    abs_url=abs_url,
                    comment=comment,
                    journal_ref=journal_ref,
                    doi=doi
                )
                papers.append(paper)
                
        except ET.ParseError as e:
            raise ArxivAPIError(f"Failed to parse arXiv response: {e}")
        
        return papers
    
    async def search_papers(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        max_results: int = 100,
        start: int = 0,
        sort_by: str = "relevance",  # relevance, lastUpdatedDate, submittedDate
        sort_order: str = "descending"
    ) -> Tuple[List[ArxivPaper], int]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query (supports arXiv query syntax)
            categories: Optional list of arXiv categories to filter
            max_results: Maximum number of results to return (max 2000 per query)
            start: Starting index for pagination
            sort_by: Sort field
            sort_order: ascending or descending
            
        Returns:
            Tuple of (list of ArxivPaper, total results available)
        """
        await self._rate_limit()
        
        # Build search query
        search_terms = []
        
        # Main text query - use AND for each word instead of exact phrase match
        # This gives MUCH better results on arXiv API
        if query:
            clean_query = query.strip()
            # Split into words and search with AND between them (not exact phrase)
            words = clean_query.split()
            if len(words) > 1:
                # Search each word with AND: (all:word1 AND all:word2 AND ...)
                word_terms = " AND ".join([f"all:{word}" for word in words])
                search_terms.append(f"({word_terms})")
            else:
                search_terms.append(f"all:{clean_query}")
        
        # Add category filters
        if categories:
            cat_terms = " OR ".join([f"cat:{cat}" for cat in categories])
            if cat_terms:
                search_terms.append(f"({cat_terms})")
        
        # Combine search terms
        search_query = " AND ".join(search_terms) if search_terms else "all:*"
        
        # Build API URL
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": min(max_results, MAX_RESULTS_PER_QUERY),
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        
        print(f"\n🔎 Searching arXiv: {query}")
        print(f"   Categories: {categories or 'All'}")
        print(f"   Max results: {max_results}, Starting from: {start}")
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise ArxivAPIError(f"arXiv API returned status {response.status}")
                
                xml_content = await response.text()
                
            papers = self._parse_arxiv_response(xml_content)
            
            # Try to get total results from feed
            # arXiv includes opensearch:totalResults in response
            total_results = len(papers)  # Fallback
            try:
                root = ET.fromstring(xml_content)
                total_el = root.find('.//{http://a9.com/-/spec/opensearch/1.1/}totalResults')
                if total_el is not None:
                    total_results = int(total_el.text)
            except:
                pass
            
            self.stats["searched"] += 1
            self.stats["found"] += len(papers)
            
            print(f"   📚 Found {len(papers)} papers (Total available: {total_results})")
            
            return papers, total_results
            
        except aiohttp.ClientError as e:
            raise ArxivAPIError(f"Network error during arXiv search: {e}")
    
    async def download_pdf(self, paper: ArxivPaper, save_folder: str = None) -> Optional[str]:
        """
        Download a paper's PDF.
        
        Args:
            paper: ArxivPaper object
            save_folder: Folder to save PDF (defaults to DOWNLOAD_FOLDER)
            
        Returns:
            Path to saved file, or None if failed
        """
        save_folder = save_folder or DOWNLOAD_FOLDER
        os.makedirs(save_folder, exist_ok=True)
        
        filepath = os.path.join(save_folder, paper.clean_filename)
        
        # Skip if already exists
        if os.path.exists(filepath):
            print(f"   ⏭️ Already exists: {paper.clean_filename[:60]}")
            return filepath
        
        # Rate limit for PDF downloads
        await asyncio.sleep(PDF_DOWNLOAD_DELAY)
        
        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.get(paper.pdf_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Verify it's a PDF
                        if not content.startswith(b'%PDF'):
                            print(f"   ⚠️ Not a valid PDF: {paper.arxiv_id}")
                            return None
                        
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(content)
                        
                        return filepath
                    
                    elif response.status == 429:
                        # Rate limited - wait and retry
                        wait_time = 30 * (attempt + 1)
                        print(f"   ⏳ Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        
                    else:
                        print(f"   ❌ HTTP {response.status} for {paper.arxiv_id}")
                        if attempt == MAX_RETRIES - 1:
                            return None
                            
            except asyncio.TimeoutError:
                print(f"   ⏳ Timeout, retrying... ({attempt + 1}/{MAX_RETRIES})")
            except Exception as e:
                print(f"   ❌ Download error: {e}")
                if attempt == MAX_RETRIES - 1:
                    return None
        
        return None
    
    def _check_duplicate(self, paper: ArxivPaper) -> bool:
        """Check if paper is already in memory."""
        if not self.memory:
            return False
        
        try:
            result = self.memory.check_paper_duplicate(paper.arxiv_id, paper.title)
            return result.get("is_duplicate", False)
        except Exception as e:
            print(f"   ⚠️ Duplicate check failed: {e}")
            return False
    
    def _add_to_memory(self, paper: ArxivPaper) -> bool:
        """Add paper to shared memory."""
        if not self.memory:
            return False
        
        try:
            return self.memory.add_paper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors_str,
                abstract=paper.abstract[:500],  # Truncate for storage
                categories=",".join(paper.categories),
                downloaded_by=self.user_name
            )
        except Exception as e:
            print(f"   ⚠️ Failed to add to memory: {e}")
            return False
    
    def _save_to_json(self, paper: ArxivPaper, filepath: str):
        """Save paper metadata to JSON file."""
        try:
            with open(RESOURCES_FILE, "r+") as f:
                data = json.load(f)
                paper_dict = paper.to_dict()
                paper_dict["local_path"] = filepath
                paper_dict["downloaded_by"] = self.user_name
                paper_dict["downloaded_at"] = datetime.now().isoformat()
                data["papers"].append(paper_dict)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            print(f"   ⚠️ Failed to save metadata: {e}")
    
    async def collect_papers(
        self,
        query: str,
        max_papers: int = 100,
        categories: Optional[List[str]] = None,
        save_folder: str = None,
        skip_duplicates: bool = True
    ) -> Tuple[str, int, List[Dict]]:
        """
        Main method to collect papers from arXiv.
        
        Args:
            query: Search query
            max_papers: Maximum papers to download
            categories: Optional category filter
            save_folder: Where to save PDFs
            skip_duplicates: Whether to skip papers already in memory
            
        Returns:
            Tuple of (status message, papers downloaded, list of paper info)
        """
        save_folder = save_folder or DOWNLOAD_FOLDER
        downloaded_papers = []
        total_downloaded = 0
        start_index = 0
        
        print(f"\n{'='*60}")
        print(f"🎯 arXiv Collection: {query}")
        print(f"   Target: {max_papers} papers")
        print(f"   Categories: {categories or 'All'}")
        print(f"{'='*60}")
        
        while total_downloaded < max_papers:
            # Calculate how many to fetch this batch
            remaining = max_papers - total_downloaded
            batch_size = min(remaining, MAX_RESULTS_PER_QUERY)
            
            try:
                papers, total_available = await self.search_papers(
                    query=query,
                    categories=categories,
                    max_results=batch_size,
                    start=start_index,
                    sort_by="submittedDate",
                    sort_order="descending"
                )
            except ArxivAPIError as e:
                print(f"❌ Search failed: {e}")
                break
            
            if not papers:
                print("   📭 No more papers found")
                break
            
            # Process each paper
            for paper in papers:
                if total_downloaded >= max_papers:
                    break
                
                # Check for duplicates
                if skip_duplicates and self._check_duplicate(paper):
                    print(f"\n⏭️ SKIPPING (duplicate): {paper.title[:50]}...")
                    self.stats["skipped_duplicate"] += 1
                    continue
                
                print(f"\n📄 [{total_downloaded + 1}/{max_papers}] {paper.title[:60]}...")
                print(f"   👤 {paper.authors_str[:50]}")
                print(f"   🏷️ {paper.primary_category}")
                print(f"   📅 {paper.published[:10]}")
                
                # Download PDF
                filepath = await self.download_pdf(paper, save_folder)
                
                if filepath:
                    print(f"   💾 Saved: {paper.clean_filename[:50]}...")
                    
                    # Save to JSON and memory
                    self._save_to_json(paper, filepath)
                    self._add_to_memory(paper)
                    
                    downloaded_papers.append({
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": paper.authors_str,
                        "filename": paper.clean_filename
                    })
                    
                    total_downloaded += 1
                    self.stats["downloaded"] += 1
                else:
                    print(f"   ❌ Failed to download")
                    self.stats["failed"] += 1
            
            # Move to next page
            start_index += len(papers)
            
            # Check if we've exhausted results
            if start_index >= total_available:
                print(f"\n📚 Exhausted all {total_available} available papers")
                break
        
        # Summary
        print(f"\n{'='*60}")
        print(f"✅ Collection Complete!")
        print(f"   📥 Downloaded: {total_downloaded} papers")
        print(f"   ⏭️ Skipped (duplicates): {self.stats['skipped_duplicate']}")
        print(f"   ❌ Failed: {self.stats['failed']}")
        print(f"{'='*60}")
        
        self.downloaded_papers.extend(downloaded_papers)
        
        return (
            f"Successfully downloaded {total_downloaded} papers about '{query}'",
            total_downloaded,
            downloaded_papers
        )
    
    async def bulk_collect(
        self,
        topics: List[Dict[str, Any]],
        total_target: int = 5000
    ) -> Dict:
        """
        Collect papers across multiple topics for large-scale dataset building.
        
        Args:
            topics: List of dicts with 'query', 'categories', 'weight' keys
            total_target: Total number of papers to collect
            
        Returns:
            Summary statistics
        """
        print(f"\n{'='*70}")
        print(f"🚀 BULK COLLECTION: Targeting {total_target} papers across {len(topics)} topics")
        print(f"{'='*70}")
        
        # Calculate papers per topic based on weight
        total_weight = sum(t.get('weight', 1) for t in topics)
        
        all_papers = []
        topic_stats = {}
        
        for topic in topics:
            query = topic['query']
            categories = topic.get('categories')
            weight = topic.get('weight', 1)
            
            # Calculate target for this topic
            topic_target = int((weight / total_weight) * total_target)
            topic_target = max(topic_target, 10)  # Minimum 10 papers per topic
            
            print(f"\n📌 Topic: {query} (target: {topic_target} papers)")
            
            message, count, papers = await self.collect_papers(
                query=query,
                max_papers=topic_target,
                categories=categories
            )
            
            all_papers.extend(papers)
            topic_stats[query] = count
        
        print(f"\n{'='*70}")
        print(f"🎉 BULK COLLECTION COMPLETE!")
        print(f"   📊 Total papers: {len(all_papers)}")
        print(f"   📈 By topic:")
        for topic, count in topic_stats.items():
            print(f"      - {topic}: {count}")
        print(f"{'='*70}")
        
        return {
            "total_downloaded": len(all_papers),
            "by_topic": topic_stats,
            "papers": all_papers
        }


# --- UTILITY FUNCTIONS ---

def get_common_categories() -> Dict[str, List[str]]:
    """Get predefined category groups for common ML/AI research areas."""
    return {
        "nlp": [ArxivCategory.CS_CL.value, ArxivCategory.CS_AI.value],
        "computer_vision": [ArxivCategory.CS_CV.value],
        "machine_learning": [ArxivCategory.CS_LG.value, ArxivCategory.STAT_ML.value],
        "deep_learning": [ArxivCategory.CS_LG.value, ArxivCategory.CS_NE.value],
        "reinforcement_learning": [ArxivCategory.CS_LG.value, ArxivCategory.CS_AI.value],
        "robotics": [ArxivCategory.CS_RO.value],
        "all_cs_ai": [
            ArxivCategory.CS_AI.value,
            ArxivCategory.CS_CL.value,
            ArxivCategory.CS_CV.value,
            ArxivCategory.CS_LG.value,
            ArxivCategory.CS_NE.value
        ]
    }


async def core_arxiv_download_logic(
    query: str,
    max_papers: int = 100,
    categories: Optional[List[str]] = None,
    memory=None,
    user_name: str = "unknown"
) -> Tuple[str, int, List[Dict]]:
    """
    Core function for arXiv paper download, callable by agent.py.
    
    This is the main entry point for the agent to collect papers.
    """
    async with ArxivCollector(memory=memory, user_name=user_name) as collector:
        return await collector.collect_papers(
            query=query,
            max_papers=max_papers,
            categories=categories
        )


# --- TEST/DEMO ---
if __name__ == "__main__":
    async def demo():
        """Demo the arXiv collector."""
        print("\n" + "="*60)
        print("🔬 arXiv Collector Demo")
        print("="*60)
        
        async with ArxivCollector(user_name="demo_user") as collector:
            # Search for transformers papers
            message, count, papers = await collector.collect_papers(
                query="transformer neural network",
                max_papers=5,
                categories=[ArxivCategory.CS_CL.value, ArxivCategory.CS_LG.value]
            )
            
            print(f"\n{message}")
            print(f"\nDownloaded papers:")
            for p in papers:
                print(f"  - {p['title'][:60]}...")
    
    asyncio.run(demo())
