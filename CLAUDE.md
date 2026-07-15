# CLAUDE.md — Claude Code layer for CSX4201/Project

> **Relationship to sibling files (they complement each other)**
> - `AGENTS.md` is the **shared, cross-host** source of truth (Codex + Claude + Gemini). Shared rules, project identity, dataset structure, session-start protocol, privacy, and pending items all live there.
> - This file (`CLAUDE.md`) adds **Claude Code-specific** behavior only. It does not duplicate `AGENTS.md`.
> - **Conflict rule for Claude Code:** `CLAUDE.md` > `AGENTS.md` at equal scope; both defer to the user's global `~/.claude/CLAUDE.md`.
> - Read `AGENTS.md` for anything not covered here.

## At session start (MUST)
1. Read `AGENT_MEMORY.md` — verify any fact against the live filesystem before relying on it; reading is orientation, not completion.
2. Read `LESSONS.md` — do not repeat recorded mistakes.
3. Re-read whichever section of `AGENTS.md` is relevant to the current task (especially *Privacy* before any commit/share).

## Working style for this project (Claude Code specifics)
- Prefer dedicated tools (`Read`, `Glob`, `Grep`, `Edit`) over shell `cat`/`grep`/`find`. The dataset folders are large; use narrow globs and shallow `find` depths to avoid timeouts.
- The data root contains many `.zip` archives. Do not unzip on assumption — ask first; unzipping can explode working-tree size and complicate the later GitHub upload.
- When proposing file changes, keep diffs minimal and match existing conventions (none established yet in this repo — keep it plain and simple).
- Use `TodoWrite` for multi-step work. Mark items in_progress one at a time and complete them as finished.
- For multi-file repetitive work, delegate via the Agent tool per the user's global agent-delegation rules; retain architecture, security, and final validation in the main agent.

## Privacy guardrail (Claude Code enforcement)
- Before running any `git add`, `git commit`, `git push`, or any tool that uploads content externally, re-check `AGENTS.md` *Privacy*. The `gmail_private_test/` folder is gitignored by default and MUST stay excluded unless the user explicitly opts in to a private repo.
- Treat the *No Emojis* and *English-only* global rules as hard constraints in all file contents and responses.

## Skills posture
- The user maintains a large skills catalog. If a skill clearly applies (e.g., a notebook-execution, data-visualization, or academic-writing skill), invoke it; otherwise proceed normally. Do not force-fit skills that do not match the task.

## Definition of done
- The requested scope is finished for the active task type.
- Critical risks and edge cases that materially affect the result are addressed.
- Reports state explicitly what was verified vs. only reviewed, and flag anything skipped or blocked.
