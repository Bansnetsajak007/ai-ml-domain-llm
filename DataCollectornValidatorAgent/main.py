#!/usr/bin/env python3
"""
RAMESH - AI-Powered Data Collector Agent v2.0
==============================================
Main entry point for the RAMESH agent.

Usage:
    python main.py              # Start interactive agent
    python main.py --help       # Show help
    
For bulk collection:
    python scripts/run_curated_collection.py --dry-run
    python scripts/run_curated_collection.py --resume

Created by Sajak 🇳🇵
"""

import os
import sys
import asyncio

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load environment from config folder
import dotenv
config_path = os.path.join(PROJECT_ROOT, 'config', '.env')
dotenv.load_dotenv(config_path)


def main():
    """Main entry point."""
    from src.core.agent import main as agent_main
    asyncio.run(agent_main())


if __name__ == "__main__":
    main()
