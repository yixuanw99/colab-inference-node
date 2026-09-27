#!/usr/bin/env bash
# ==============================================================================
# Colab Model Station - Automated Non-Interactive Git Synchronization Tool
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

MSG="${1:-}"

if [ -z "$MSG" ]; then
    echo "Usage: bash scripts/sync_git.sh '<conventional-commit-message>'"
    echo "Example: bash scripts/sync_git.sh 'feat(diffusers): add fp8 flux support'"
    exit 1
fi

# Validate Conventional Commits pattern
PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-zA-Z0-9_-]+\))?: .+"
if [[ ! "$MSG" =~ $PATTERN ]]; then
    echo "[ERROR] Commit message does not adhere to Conventional Commits standard:"
    echo "  Provided: '$MSG'"
    echo "  Expected: <type>(<optional scope>): <description>"
    exit 1
fi

echo "[GIT] Staging all tracked and untracked changes..."
git add -A

if git diff-index --quiet HEAD --; then
    echo "[GIT] Working tree is clean. Nothing to commit."
    exit 0
fi

echo "[GIT] Committing changes with message: '$MSG'..."
git commit -m "$MSG"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "[GIT] Pushing commits to remote origin/$BRANCH..."
git push origin "$BRANCH"

echo "[GIT] Synchronization complete."
