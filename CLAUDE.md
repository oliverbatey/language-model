# Project: language-model

## What this is
This is an implementation of a transformer-based language model from scratch. This means you are not permitted to use definitions from `torch.nn`, `torch.nn.functional` or `torch.optim` except:
- `torch.nn.parameter`
- Container classes in `torch.nn`, e.g. `Module`, `ModuleList`, `Sequential`, etc
- The `torch.optim.Optimizer` base class
Other deinititions from PyTorch are allowed.

## Persistent context
My notes repo lives at ~/Repos/notes/

At the start of every session:
- Read ~/Repos/notes/projects/language-model/status.md for current
  status, open items, and blockers

When I say "wrap up" or the session ends naturally:
- Update ~/Repos/notes/projects/language-model/status.md with any
  status changes, new decisions, or resolved items
- Append a brief summary to ~/Repos/notes/journal/2026.md in this format:
  ## YYYY-MM-DD
  - [what was done]
  - [decisions made]
  - [next steps]

## Worktree workflow
All new features and fixes must be done in a git worktree, not main.
Before starting any coding task:
1. Ask me for a branch name if I haven't given one
2. Tell me the exact command to run:
   git worktree add -b <branch-name> ../language-model-<branch-name> main
3. Confirm which directory you are working in before editing any files
4. Stay within that directory — do not edit files outside it
When the task is complete, remind me to commit, push, and remove the worktree.

## Coding conventions
- Clear, readable code over clever code
- Every function needs a docstring
- Explicit error handling — no silent failures or bare except clauses
- Write tests alongside new features, not after
- Keep functions small and single-purpose

## Git conventions
- Commit messages: imperative, lowercase, under 72 chars
- One logical change per commit
- Never commit: API keys, .env files, credentials of any kind

## Pre-push checklist
Before any git push, confirm:
1. Tests pass
2. No hardcoded credentials or API keys
3. .env is in .gitignore
4. Commit message follows convention

## What to avoid
- Do not modify pyproject.toml directly — tell me what to add
- Do not delete files without confirming with me first
- Do not make architectural decisions silently — flag and ask
- Do not push to main directly