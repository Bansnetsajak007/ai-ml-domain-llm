# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-01-25

### 🔴 Critical Fixes

#### Fixed
- **[CRITICAL]** JSON file corruption in `mcp_server.py::save_to_json_file()`
  - Added `f.truncate()` to prevent trailing data corruption
  - Added JSON corruption recovery mechanism
  - Impact: Prevents data loss in resources.json

- **[CRITICAL]** Dual memory system causing sync issues
  - Removed SQLite local database completely
  - Now uses Supabase cloud exclusively
  - Eliminated `is_duplicate()` function that only checked local DB
  - Impact: True shared memory across all team members

- **[CRITICAL]** Browser resource leaks
  - Added proper `finally` blocks for cleanup
  - Browsers now close even on errors
  - Impact: Prevents zombie processes consuming memory

- **[CRITICAL]** Unconditional 40-second delays
  - Delays now only apply after successful downloads
  - Made configurable via `DOWNLOAD_COOLDOWN` env var
  - Impact: Saves time on failed downloads

### 🟠 Major Improvements

#### Added
- Environment variable validation at startup
  - Checks for required: `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`
  - Clear error messages with setup instructions
  
- LLM timeout protection
  - All OpenAI API calls now have 30s timeout (configurable)
  - Graceful fallbacks on timeout
  
- Retry logic with exponential backoff
  - Up to 3 retries for failed LLM calls
  - Exponential backoff: 1s, 2s, 4s
  
- Conversation history management
  - Auto-trim to prevent memory leak
  - Configurable via `MAX_CONVERSATION_HISTORY` (default: 20)
  
- MongoDB duplicate key error handling
  - Gracefully catches `DuplicateKeyError`
  - No more crashes on duplicate insertions
  
- Input sanitization
  - User names sanitized (alphanumeric + safe chars)
  - Limited to 50 characters

#### Changed
- Browser headless mode now configurable
  - Set `BROWSER_HEADLESS=true` for production
  - Default: false (for debugging)
  
- All magic numbers moved to environment variables
  - `MAX_DOWNLOADS_PER_ACCOUNT` (default: 9)
  - `DOWNLOAD_COOLDOWN` (default: 40)
  - `LLM_TIMEOUT` (default: 30)
  
- Improved error handling
  - Specific exception catching
  - Better error messages
  - Proper resource cleanup

### 📝 Documentation

#### Added
- `.env.example` for DataCollectornValidatorAgent
- `.env.example` for WebDataTracker
- Comprehensive changelog in README.md
- Migration guide for breaking changes
- Testing recommendations

### ⚠️ Breaking Changes

- **SQLite database removed**
  - `shared_memory.db` no longer used
  - Must use Supabase for shared memory
  - Old SQLite data not migrated
  - **Action Required**: Set up Supabase credentials

- **Environment variables now required**
  - Script exits if missing required vars
  - **Action Required**: Create `.env` from `.env.example`

### 🗑️ Removed
- `sqlite3` import from `mcp_server.py`
- `log_download_to_db()` function (used SQLite)
- `is_duplicate()` function (used SQLite)
- `DB_PATH` configuration variable
- Debug code and excessive screenshots

### 📦 Dependencies

#### No changes to requirements.txt
- Still using: fastmcp, playwright, openai, python-dotenv, postgrest, numpy
- SQLite was stdlib, so no package removal needed

### 🐛 Bug Fixes

- Fixed potential SQL injection in greeting (sanitized input)
- Fixed missing error handling in account loading
- Fixed browser not closing on page load errors
- Fixed conversation list growing unbounded
- Fixed no timeout on network calls

### 🧪 Testing

Recommended tests before merge:
- [ ] Test with missing environment variables
- [ ] Test download flow with multiple accounts  
- [ ] Verify Supabase duplicate detection
- [ ] Test conversation trimming
- [ ] Verify browser cleanup (no zombies)
- [ ] Test WebDataTracker duplicate handling

### 📊 Impact Summary

| Category | Before | After | Impact |
|----------|--------|-------|--------|
| Data Loss Risk | High (JSON corruption) | Low | 🔴 Critical |
| Memory Sync | Broken (dual system) | Working (Supabase only) | 🔴 Critical |
| Resource Leaks | Yes (browsers) | No (proper cleanup) | 🔴 Critical |
| Wasted Time | 40s per failure | 0s per failure | 🟠 Major |
| Memory Leak | Yes (conversation) | No (auto-trim) | 🟠 Major |
| Network Hangs | Possible (no timeout) | Prevented (30s timeout) | 🟠 Major |
| Configuration | Hardcoded | Environment vars | 🟢 Good |

---

## Legend

- 🔴 Critical: Data loss, corruption, or system failure
- 🟠 Major: Significant bugs or improvements
- 🟡 Moderate: Nice to have improvements
- 🟢 Minor: Code quality, documentation
