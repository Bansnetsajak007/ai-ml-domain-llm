# RAMESH Codebase Review and Production Upgrade Plan

Date: 2026-01-30
Author: GitHub Copilot (AI Engineer)

Overview
- Goal: Make Ramesh reliable, maintainable, and production-ready.
- Scope: Architecture, reliability gaps, login/rotation, download workflow, metadata/memory, observability, configuration, security, testing, and roadmap.

Executive Summary
- Strengths: Clear agent orchestration, Playwright automation, Supabase shared memory, simple linear account rotation, LLM metadata extraction.
- Pain Points: Login state fragile, rotation doesn’t verify login per account, daily-limit detection scattered, download flow brittle, limited observability, no tests/CI.
- Quick Wins: Switch to Playwright storage-state login; add `ensure_logged_in()`; centralize rotation; harden daily-limit detection; structured logging; normalize metadata; add file hashing; basic tests.

Current Architecture
- Agent CLI: agent.py (conversation, tools, session state, rotation, memory usage).
- Automation: mcp_server.py (search + download, cookie login, metadata, JSON logging, memory writes).
- Memory: memory.py (Supabase PostgREST, embeddings, duplicate checks, stats).
- Utilities: run.py (manual orchestrator), check_site_status.py, setup_db.py (legacy local DB).

Key Reliability Gaps
1) Login Fragility: Cookie injection often fails; no canonical “am I logged in?” verification.
2) Account Rotation: Linear, but no login verification after switch; no persisted limit state.
3) Daily Limit Detection: Inline; needs consolidated detector.
4) Download Workflow: Brittle to UI changes; limited retries/backoff.
5) Observability: Print logs only; no structured fields or failure artefacts.
6) Configuration & Secrets: Stale cookies; no storage-state files; limited validation.
7) Testing & CI: No unit/integration tests; no smoke test.

Production-Grade Improvements
A) Authentication & Session
- Use Playwright storage-state per account.
- Add `ensure_logged_in(page)` to verify “My Library”/no “Log In”.

B) Account Manager
- New `account_manager.py` with linear queue, persisted state (`data/account_state.json`), apply storage-state contexts, mark limits.

C) Daily Limit Detector
- `is_daily_limit_page(page)` checks: page text, promo panel, donation CTA, URL patterns, deterministic selectors.

D) Download Flow Hardening
- Navigate to detail + click explicit download anchors; fallback to href.
- Add retries with exponential backoff; mirror selection for PDF.

E) Metadata & Normalization
- Normalize title/authors; store `normalized_title`, canonical author list.
- Persist `source_url`, `extension`, `size`, `sha256` hash.

F) Supabase Memory Enhancements
- Index `(normalized_title, authors)`.
- Batch insert with 429 retries; embedding cache; richer stats.

G) Observability & Debug
- Structured JSON logs; fields: account_id, topic, book_id, action, outcome, duration, error_code.
- Save failure screenshots under `debug/{account}/{topic}/{step}.png`.
- Emit session summary JSON.

H) Configuration & Secrets
- Storage-state JSONs under `accounts/` (gitignored).
- Env flags: `LOGIN_METHOD`, `PROXY_ENABLED`, `PROXY_URL`.
- Startup validation for storage-state paths.

I) Reliability & Resilience
- Adaptive cooldowns; optional proxy pool; idempotent downloads (filename/hash check).

J) Testing & CI
- Unit tests for login checks, normalization, limit detector.
- Playwright smoke test to verify login; GitHub Actions CI.

Feature Roadmap
- Topic planner; resumable sessions; export catalog; UI change alerts; cross-account duplicate suppression.

Suggested Modules (Sketch)
- account_manager.py: `current()`, `next()`, `apply_context(browser)`, `mark_limit_reached()`.
- auth_utils.py: `ensure_logged_in(page)`, `is_daily_limit_page(page)`.
- download_utils.py: `click_primary_download(page)`, `retry_with_backoff(...)`.

Phased Implementation
Phase 1 (Stability): storage-state login, ensure_logged_in, centralized rotation, consolidated limit detector.
Phase 2 (Resilience): hardened download flow, normalization + hashing, structured logging + screenshots.
Phase 3 (Ops & Scale): tests + CI, proxy support, topic planner, resumable sessions.

Runbook
- Capture storage-states per account; set `LOGIN_METHOD=storage_state`.
- Headed smoke test verifies “My Library” present before search.

Closing Notes
- Storage-state login + explicit verification yields the largest reliability gain.
- Recommend implementing Phase 1 first; I can proceed once storage-state files are captured.
