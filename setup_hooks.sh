#!/bin/bash
# Setup script for Git hooks
# Created: October 24, 2025
# Last Updated: October 24, 2025
#
# This script configures Git to use the custom hooks in .githooks/
# Run this once after cloning the repository

set -e

echo "🔧 Setting up Git hooks for Bond Pipeline Project..."
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ ERROR: Not in a Git repository"
    echo "   Run this script from the project root"
    exit 1
fi

# Check if .githooks directory exists
if [[ ! -d ".githooks" ]]; then
    echo "❌ ERROR: .githooks directory not found"
    echo "   Make sure you're in the project root"
    exit 1
fi

# Configure Git to use .githooks directory
echo "📁 Configuring Git to use .githooks directory..."
git config core.hooksPath .githooks
echo "✅ Git configured to use .githooks"
echo ""

# Verify configuration
hooks_path=$(git config core.hooksPath)
if [[ "$hooks_path" == ".githooks" ]]; then
    echo "✅ Verification successful"
    echo "   Hooks path: $hooks_path"
else
    echo "⚠️  WARNING: Hooks path may not be set correctly"
    echo "   Expected: .githooks"
    echo "   Got: $hooks_path"
fi
echo ""

# Make hooks executable (just in case)
echo "🔐 Ensuring hooks are executable..."
chmod +x .githooks/*
echo "✅ Hooks are executable"
echo ""

# List available hooks
echo "📋 Available hooks:"
for hook in .githooks/*; do
    if [[ -f "$hook" ]] && [[ -x "$hook" ]]; then
        hook_name=$(basename "$hook")
        echo "   ✅ $hook_name"
    fi
done
echo ""

# Test hooks
echo "🧪 Testing hooks..."
echo ""

# Test pre-commit hook
echo "   Testing pre-commit hook..."
if .githooks/pre-commit 2>&1 | grep -q "pre-commit checks"; then
    echo "   ✅ pre-commit hook works"
else
    echo "   ⚠️  pre-commit hook may have issues"
fi

# Test commit-msg hook
echo "   Testing commit-msg hook..."
echo "Test commit message" > /tmp/test_commit_msg.txt
if .githooks/commit-msg /tmp/test_commit_msg.txt 2>&1 | grep -q "Commit message valid"; then
    echo "   ✅ commit-msg hook works"
else
    echo "   ⚠️  commit-msg hook may have issues"
fi
rm -f /tmp/test_commit_msg.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Git hooks setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📚 Next steps:"
echo "   1. Make a test commit to verify hooks work"
echo "   2. Read Documentation/Reference/Git-Hooks-Guide.md for details"
echo "   3. To bypass hooks (emergency only): git commit --no-verify"
echo ""
echo "🎉 Happy coding!"

