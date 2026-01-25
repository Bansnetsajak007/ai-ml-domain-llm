# AI/ML Domain LLM

An engineering-focused project covering data acquisition, dataset curation, tokenization, training, and evaluation of a domain-specific AI/ML LLM.

## Project Goal

The primary goal of this project is to train a domain-specific Large Language Model (LLM) from scratch, focused on Artificial Intelligence and Machine Learning knowledge.

This repository documents and implements the entire LLM pipeline end-to-end, with a strong emphasis on engineering clarity rather than speed or scale.

**This is not about using pre-trained APIs. This is about understanding and building the system.**

## Scope of the Project

This repository covers:

### 1. Data Acquisition
- Automated and semi-automated collection of AI/ML-related books and resources
- Topic-driven data gathering (subdomains within AI/ML)

### 2. Dataset Curation
- Metadata generation
- Duplicate detection
- Content validation and filtering

### 3. Tokenization
- Text normalization
- Vocabulary construction
- Token statistics and analysis

### 4. Model Training
- Training a language model from scratch
- Domain-focused learning objectives

### 5. Evaluation
- Domain-specific evaluation prompts
- Qualitative and quantitative assessment

### 6. Production Deployment
- Model optimization and quantization
- Inference pipeline setup
- API endpoint design and implementation
- Performance monitoring and logging

## System Design Philosophy

- Engineering-first approach
- Automation where possible, manual control where necessary
- Slow is acceptable, correctness is not optional
- Reproducibility over convenience
- Production-ready code from the start

The system is designed to reduce manual effort while keeping the learning process transparent and debuggable.

## Repository Structure (Planned)

```
ai-ml-domain-llm/
│
├── data-collection/      # Agents, tools, metadata generators
├── data-checking/        # Deduplication, validation, statistics
├── datasets/             # Curated and processed datasets
├── tokenization/         # Tokenizer experiments and analysis
├── training/             # Model architecture and training scripts
├── evaluation/           # Evaluation prompts and metrics
├── deployment/           # Production deployment configurations
├── docs/                 # Architecture and design notes
└── README.md
```

## Current Status

- Project initialization
- Data collection architecture design
- Tooling for metadata and duplicate checking

The repository will evolve incrementally as each stage of the pipeline is implemented.

## Team

This project is developed collaboratively by a small team, with shared responsibility across system design, data engineering, and model experimentation.

## Disclaimer

This project is intended for educational and research purposes only. All data handling is performed with respect to learning objectives and system design exploration.

## Why This Project Exists

Most people use LLMs. Few understand how they are built and deployed.

This project exists to gain real, hands-on experience with LLM engineering—from data collection through training to production deployment. It's an end-to-end journey through the complete lifecycle of building and shipping a language model.

**From raw text → tokens → parameters → behavior → production.**

---

## 🔧 Recent Updates (Branch: `bugfix/critical-fixes`)

### Critical Fixes Applied ✅

**Date**: January 25, 2026

This branch contains comprehensive fixes for critical bugs and improvements to code quality, reliability, and production-readiness.

#### 🔴 Critical Issues Fixed

1. **JSON File Corruption Prevention**
   - Fixed race condition in `save_to_json_file()` that could corrupt `resources.json`
   - Added proper `f.truncate()` after writing
   - Added recovery mechanism for corrupted JSON files

2. **Unified Memory System**
   - **REMOVED** SQLite local database (`shared_memory.db`)
   - Now using **Supabase cloud exclusively** for shared memory
   - Eliminated dual memory system that was causing sync issues
   - All team members now see the same duplicate detection data

3. **Browser Resource Management**
   - Added proper cleanup in `finally` blocks to prevent zombie browser processes
   - Browsers now always close even on errors
   - Configurable headless mode via `BROWSER_HEADLESS` environment variable

4. **Conditional Download Delays**
   - Download cooldown (40s) now only applies to successful downloads
   - Configurable via `DOWNLOAD_COOLDOWN` environment variable
   - Failures no longer waste time waiting

#### 🟠 Major Improvements

5. **Environment Variable Validation**
   - Added startup checks for required variables (`OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`)
   - Clear error messages guide users to fix missing configuration
   - Prevents cryptic runtime errors

6. **LLM Timeout Protection**
   - All OpenAI API calls now have configurable timeouts (default: 30s)
   - Prevents indefinite hangs on network issues
   - Graceful fallbacks when LLM calls fail

7. **Conversation Memory Leak Fix**
   - Added automatic conversation history trimming (default: 20 messages)
   - Prevents token overflow in long sessions
   - Configurable via `MAX_CONVERSATION_HISTORY`

8. **Retry Logic with Exponential Backoff**
   - LLM calls retry up to 3 times on failure
   - Exponential backoff (1s, 2s, 4s) prevents hammering failed endpoints
   - User-friendly error messages

9. **MongoDB Duplicate Key Handling**
   - Gracefully handles unique index violations in WebDataTracker
   - No more crashes on duplicate insertions
   - Proper error counting and reporting

10. **Input Sanitization**
    - User names are now sanitized (alphanumeric + basic chars only)
    - Limited to 50 characters
    - Prevents potential injection issues

#### 🟢 Code Quality Improvements

- **Configuration Management**: All magic numbers moved to environment variables
- **Error Handling**: Specific exception catching instead of broad `except Exception`
- **Resource Cleanup**: Proper context management and cleanup in error cases
- **.env.example Files**: Added template files for both DataCollector and WebDataTracker
- **Type Safety**: Added validation for loaded JSON configurations

### Configuration Changes

New environment variables (see `.env.example` files):
- `BROWSER_HEADLESS` - Set to `true` for production servers
- `DOWNLOAD_COOLDOWN` - Adjust download rate limiting
- `LLM_TIMEOUT` - Prevent hanging on slow API responses
- `MAX_DOWNLOADS_PER_ACCOUNT` - Adjust for Z-Library limit changes
- `MAX_CONVERSATION_HISTORY` - Control memory usage in long sessions

### Breaking Changes ⚠️

- **Removed SQLite dependency**: `shared_memory.db` is no longer used
  - **Action Required**: Ensure Supabase credentials are in `.env`
  - Old SQLite data will not be migrated automatically
- **Environment variables now required**: Script will exit if missing
  - **Action Required**: Copy `.env.example` to `.env` and fill in values

### Testing Recommendations

Before merging to main:
1. ✅ Test with missing environment variables
2. ✅ Test download flow with multiple accounts
3. ✅ Verify Supabase duplicate detection works across team members
4. ✅ Test conversation trimming in long sessions
5. ✅ Verify browser cleanup (check for zombie processes)
6. ✅ Test WebDataTracker duplicate handling

### Files Modified

- `DataCollectornValidatorAgent/agent.py` - Memory leak fix, validation, retries
- `DataCollectornValidatorAgent/mcp_server.py` - JSON fix, SQLite removal, browser cleanup
- `ManualDataValidator/WebDataTracker/WebDataTracker/app.py` - Duplicate key handling
- `ManualDataValidator/WebDataTracker/WebDataTracker/openai_service.py` - Timeout protection
- **NEW**: `DataCollectornValidatorAgent/.env.example`
- **NEW**: `ManualDataValidator/WebDataTracker/WebDataTracker/.env.example`

### Migration Guide

1. **Backup your data** (just in case):
   ```bash
   cp data/resources.json data/resources.json.backup
   ```

2. **Set up environment variables**:
   ```bash
   cd DataCollectornValidatorAgent
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

3. **Verify Supabase table exists** (run once):
   ```bash
   python memory.py
   # Follow instructions to create table in Supabase dashboard
   ```

4. **Test the agent**:
   ```bash
   python agent.py
   ```

---
