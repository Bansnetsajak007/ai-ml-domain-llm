# Data Collectors Package
# - arxiv_collector: arXiv paper downloads
# - mcp_server: Z-Library book downloads

from .arxiv_collector import ArxivCollector, ArxivPaper, core_arxiv_download_logic, get_common_categories
from .mcp_server import core_download_logic
