# Git Hooks Guide for Bond Pipeline Project

**Created**: October 24, 2025  
**Last Updated**: October 24, 2025  
**Purpose**: Educational guide and implementation reference for Git hooks

---

## Table of Contents

1. [What Are Git Hooks?](#what-are-git-hooks)
2. [Why Use Hooks in This Project?](#why-use-hooks-in-this-project)
3. [Recommended Hooks for Bond Pipeline](#recommended-hooks-for-bond-pipeline)
4. [Implementation Guide](#implementation-guide)
5. [Testing Hooks](#testing-hooks)
6. [Troubleshooting](#troubleshooting)

---

## What Are Git Hooks?

**Git hooks** are scripts that Git automatically executes before or after specific events like commits, pushes, and merges. They allow you to automate quality checks, enforce standards, and prevent common mistakes.

### Key Concepts:

- **Location**: `.git/hooks/` directory in your repository
- **Naming**: Hooks have specific names (e.g., `pre-commit`, `pre-push`)
- **Language**: Can be written in any scripting language (bash, Python, etc.)
- **Execution**: Automatically triggered by Git operations
- **Exit Codes**: `0` = success (continue), `non-zero` = failure (abort operation)

### Hook Types:

| Hook | Trigger | Use Case |
|------|---------|----------|
| **pre-commit** | Before commit is created | Lint code, run tests, check formatting |
| **commit-msg** | After commit message entered | Validate commit message format |
| **pre-push** | Before push to remote | Run full test suite, check for secrets |
| **post-commit** | After commit is created | Send notifications, update logs |
| **pre-rebase** | Before rebase operation | Prevent rebasing protected branches |

---

## Why Use Hooks in This Project?

For the **Bond Pipeline Project**, hooks provide:

### 1. **Code Quality Enforcement**
- Ensure all Python code is formatted consistently
- Run linters before commits
- Catch syntax errors early

### 2. **Data Integrity**
- Prevent committing large data files (Excel, parquet)
- Validate CUSIP formats in test data
- Check for sensitive information

### 3. **Testing Automation**
- Run unit tests before pushing
- Ensure tests pass before merging
- Maintain test coverage thresholds

### 4. **Documentation Standards**
- Ensure documentation is updated with timestamps
- Validate markdown formatting
- Check for broken links

### 5. **Workflow Consistency**
- Enforce commit message conventions
- Prevent pushing to main branch directly
- Ensure virtual environment is activated

---

## Recommended Hooks for Bond Pipeline

### 🔥 **Essential Hooks** (Implement First)

#### 1. **pre-commit** - Code Quality Gate
**Purpose**: Run before every commit to ensure code quality

**What it checks**:
- ✅ Python syntax errors
- ✅ Code formatting (Black, isort)
- ✅ Linting (flake8)
- ✅ No large files (> 10MB)
- ✅ No sensitive data (API keys, passwords)
- ✅ No debug statements (`print()`, `breakpoint()`)

**When to use**: Every commit

**Priority**: ⭐⭐⭐⭐⭐ **CRITICAL**

---

#### 2. **pre-push** - Test Gate
**Purpose**: Run before pushing to ensure all tests pass

**What it checks**:
- ✅ All unit tests pass
- ✅ Test coverage meets threshold (85%+)
- ✅ No failing integration tests
- ✅ Pipeline can run successfully

**When to use**: Before pushing to GitHub

**Priority**: ⭐⭐⭐⭐⭐ **CRITICAL**

---

#### 3. **commit-msg** - Message Validator
**Purpose**: Ensure commit messages follow conventions

**What it checks**:
- ✅ Minimum message length (10 chars)
- ✅ Starts with capital letter
- ✅ No trailing periods
- ✅ Optional: Follows conventional commits format

**When to use**: Every commit

**Priority**: ⭐⭐⭐⭐ **HIGH**

---

### 💡 **Recommended Hooks** (Implement Second)

#### 4. **post-commit** - Automation Helper
**Purpose**: Automate tasks after successful commit

**What it does**:
- 📝 Update changelog
- 📊 Log commit statistics
- 🔔 Send notifications (optional)

**Priority**: ⭐⭐⭐ **MEDIUM**

---

#### 5. **pre-rebase** - Safety Check
**Purpose**: Prevent dangerous rebase operations

**What it checks**:
- ⚠️ Prevent rebasing main branch
- ⚠️ Warn about public branch rebases

**Priority**: ⭐⭐ **LOW** (for solo development)

---

## Implementation Guide

### Step 1: Create Hooks Directory Structure

```bash
cd /path/to/Bond-RV-App
mkdir -p .githooks
```

**Why `.githooks` instead of `.git/hooks`?**
- `.git/hooks` is not tracked by Git
- `.githooks` can be committed and shared with team
- Configure Git to use `.githooks`:

```bash
git config core.hooksPath .githooks
```

---

### Step 2: Implement Pre-Commit Hook

**File**: `.githooks/pre-commit`

```bash
#!/bin/bash
# Pre-commit hook for Bond Pipeline Project
# Runs code quality checks before allowing commit

set -e  # Exit on first error

echo "🔍 Running pre-commit checks..."

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  WARNING: Not in a virtual environment!"
    echo "   Activate venv: source venv/bin/activate"
    # Don't fail, just warn
fi

# 1. Check for large files
echo "📦 Checking for large files..."
large_files=$(git diff --cached --name-only --diff-filter=ACM | xargs -I {} du -h {} 2>/dev/null | awk '$1 ~ /[0-9]+M/ && $1 > 10')
if [[ -n "$large_files" ]]; then
    echo "❌ ERROR: Large files detected (> 10MB):"
    echo "$large_files"
    echo "   Add to .gitignore or use Git LFS"
    exit 1
fi
echo "✅ No large files"

# 2. Check for sensitive data
echo "🔒 Checking for sensitive data..."
sensitive_patterns="(password|api_key|secret|token|private_key)"
if git diff --cached | grep -iE "$sensitive_patterns" > /dev/null; then
    echo "❌ ERROR: Potential sensitive data detected!"
    echo "   Review your changes for passwords, API keys, or secrets"
    exit 1
fi
echo "✅ No sensitive data detected"

# 3. Check Python syntax
echo "🐍 Checking Python syntax..."
python_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
if [[ -n "$python_files" ]]; then
    for file in $python_files; do
        python -m py_compile "$file" 2>&1 | grep -v "^$" && {
            echo "❌ Syntax error in $file"
            exit 1
        } || true
    done
    echo "✅ Python syntax valid"
else
    echo "ℹ️  No Python files to check"
fi

# 4. Run Black formatter check
echo "🎨 Checking code formatting (Black)..."
if command -v black &> /dev/null; then
    if [[ -n "$python_files" ]]; then
        black --check $python_files 2>&1 | grep -q "would be reformatted" && {
            echo "❌ Code formatting issues detected!"
            echo "   Run: black $python_files"
            exit 1
        } || echo "✅ Code formatting OK"
    fi
else
    echo "⚠️  Black not installed, skipping format check"
fi

# 5. Check for debug statements
echo "🐛 Checking for debug statements..."
if [[ -n "$python_files" ]]; then
    debug_found=$(git diff --cached | grep -E "^\+.*(print\(|breakpoint\(\)|pdb\.set_trace)" || true)
    if [[ -n "$debug_found" ]]; then
        echo "⚠️  WARNING: Debug statements found:"
        echo "$debug_found"
        echo "   Consider removing before commit"
        # Don't fail, just warn
    else
        echo "✅ No debug statements"
    fi
fi

# 6. Check for TODO comments in staged changes
echo "📝 Checking for new TODOs..."
new_todos=$(git diff --cached | grep -E "^\+.*TODO" || true)
if [[ -n "$new_todos" ]]; then
    echo "ℹ️  New TODO comments added:"
    echo "$new_todos"
fi

echo ""
echo "✅ All pre-commit checks passed!"
echo ""
```

**Make it executable**:
```bash
chmod +x .githooks/pre-commit
```

---

### Step 3: Implement Pre-Push Hook

**File**: `.githooks/pre-push`

```bash
#!/bin/bash
# Pre-push hook for Bond Pipeline Project
# Runs tests before allowing push

set -e

echo "🚀 Running pre-push checks..."

# 1. Run unit tests
echo "🧪 Running unit tests..."
if command -v pytest &> /dev/null; then
    pytest tests/unit/ -v --tb=short || {
        echo "❌ Unit tests failed!"
        echo "   Fix failing tests before pushing"
        exit 1
    }
    echo "✅ All unit tests passed"
else
    echo "⚠️  pytest not installed, skipping tests"
fi

# 2. Check test coverage
echo "📊 Checking test coverage..."
if command -v pytest &> /dev/null; then
    coverage=$(pytest --cov=bond_pipeline --cov-report=term-missing tests/unit/ 2>&1 | grep "TOTAL" | awk '{print $NF}' | sed 's/%//')
    if [[ -n "$coverage" ]] && [[ $(echo "$coverage < 80" | bc) -eq 1 ]]; then
        echo "⚠️  WARNING: Test coverage is ${coverage}% (target: 85%)"
        # Don't fail, just warn
    else
        echo "✅ Test coverage: ${coverage}%"
    fi
fi

# 3. Check if pipeline can run (smoke test)
echo "🔥 Running pipeline smoke test..."
if [[ -f "bond_pipeline/pipeline.py" ]]; then
    python -c "from bond_pipeline import pipeline; print('✅ Pipeline imports OK')" || {
        echo "❌ Pipeline import failed!"
        exit 1
    }
fi

echo ""
echo "✅ All pre-push checks passed!"
echo "🚀 Pushing to remote..."
echo ""
```

**Make it executable**:
```bash
chmod +x .githooks/pre-push
```

---

### Step 4: Implement Commit-Msg Hook

**File**: `.githooks/commit-msg`

```bash
#!/bin/bash
# Commit message hook for Bond Pipeline Project
# Validates commit message format

commit_msg_file=$1
commit_msg=$(cat "$commit_msg_file")

echo "📝 Validating commit message..."

# Check minimum length
if [[ ${#commit_msg} -lt 10 ]]; then
    echo "❌ ERROR: Commit message too short (minimum 10 characters)"
    echo "   Your message: '$commit_msg'"
    exit 1
fi

# Check if starts with capital letter
if ! [[ $commit_msg =~ ^[A-Z] ]]; then
    echo "⚠️  WARNING: Commit message should start with a capital letter"
    # Don't fail, just warn
fi

# Check for trailing period
if [[ $commit_msg =~ \.$ ]]; then
    echo "⚠️  WARNING: Commit message should not end with a period"
    # Don't fail, just warn
fi

# Optional: Check for conventional commits format
# Format: type(scope): description
# Example: feat(pipeline): add CUSIP validation
if [[ $commit_msg =~ ^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: ]]; then
    echo "✅ Conventional commit format detected"
fi

echo "✅ Commit message valid"
```

**Make it executable**:
```bash
chmod +x .githooks/commit-msg
```

---

### Step 5: Configure Git to Use Custom Hooks

```bash
cd /path/to/Bond-RV-App
git config core.hooksPath .githooks
```

**Verify configuration**:
```bash
git config core.hooksPath
# Should output: .githooks
```

---

## Testing Hooks

### Test Pre-Commit Hook

```bash
# 1. Create a test file with syntax error
echo "def broken_function(" > test_syntax.py
git add test_syntax.py
git commit -m "Test commit"
# Should fail with syntax error

# 2. Fix and try again
echo "def working_function():\n    pass" > test_syntax.py
git add test_syntax.py
git commit -m "Test commit"
# Should succeed

# 3. Clean up
git reset HEAD~1
rm test_syntax.py
```

### Test Pre-Push Hook

```bash
# 1. Make sure tests pass
pytest tests/unit/

# 2. Try to push
git push origin main
# Should run tests before pushing
```

### Test Commit-Msg Hook

```bash
# 1. Try short message
git commit --allow-empty -m "short"
# Should fail

# 2. Try proper message
git commit --allow-empty -m "Add feature to pipeline"
# Should succeed
```

---

## Bypassing Hooks (Emergency Use Only)

Sometimes you need to bypass hooks (e.g., work in progress):

```bash
# Skip pre-commit hook
git commit --no-verify -m "WIP: debugging"

# Skip pre-push hook
git push --no-verify
```

**⚠️ WARNING**: Only use `--no-verify` when absolutely necessary!

---

## Troubleshooting

### Hook Not Running

**Problem**: Hook doesn't execute

**Solutions**:
1. Check if hook is executable:
   ```bash
   chmod +x .githooks/pre-commit
   ```

2. Verify Git configuration:
   ```bash
   git config core.hooksPath
   ```

3. Check hook file name (no extension):
   ```bash
   ls -la .githooks/
   # Should see: pre-commit (not pre-commit.sh)
   ```

### Hook Fails Unexpectedly

**Problem**: Hook fails but you don't know why

**Solutions**:
1. Add debug output:
   ```bash
   set -x  # Add to top of hook script
   ```

2. Run hook manually:
   ```bash
   .githooks/pre-commit
   ```

3. Check exit codes:
   ```bash
   echo $?  # After running hook
   ```

### Permission Denied

**Problem**: `Permission denied` error

**Solution**:
```bash
chmod +x .githooks/*
```

### Hook Runs on Wrong Files

**Problem**: Hook checks files not in commit

**Solution**: Use `--cached` flag:
```bash
git diff --cached --name-only  # Only staged files
```

---

## Best Practices

### 1. **Keep Hooks Fast**
- Hooks should complete in < 5 seconds
- For slow checks, use pre-push instead of pre-commit
- Cache results when possible

### 2. **Make Hooks Informative**
- Use clear emoji and colors
- Explain what failed and how to fix
- Show progress for long-running checks

### 3. **Don't Block Everything**
- Use warnings for minor issues
- Only fail on critical problems
- Allow `--no-verify` escape hatch

### 4. **Test Hooks Regularly**
- Run hooks manually after changes
- Test with various scenarios
- Document expected behavior

### 5. **Share with Team**
- Commit hooks to `.githooks/`
- Document in README
- Provide setup script

---

## Advanced: Hook Manager (Optional)

For complex projects, consider using a hook manager:

### **pre-commit framework** (Recommended)

Install:
```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        language_version: python3.11
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black]
  
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/unit/, -v]
```

Install hooks:
```bash
pre-commit install
```

Run manually:
```bash
pre-commit run --all-files
```

---

## Summary: Recommended Setup for Bond Pipeline

### Phase 1: Essential (Implement Now)
1. ✅ **pre-commit** - Code quality checks
2. ✅ **pre-push** - Test suite
3. ✅ **commit-msg** - Message validation

### Phase 2: Enhanced (Implement Later)
4. ⏳ **post-commit** - Automation
5. ⏳ **pre-rebase** - Safety checks

### Phase 3: Advanced (Optional)
6. ⏳ **pre-commit framework** - Managed hooks
7. ⏳ **CI/CD integration** - GitHub Actions

---

## Next Steps

1. **Create `.githooks/` directory**
2. **Implement pre-commit hook** (start simple)
3. **Test with dummy commits**
4. **Add pre-push hook**
5. **Configure Git**: `git config core.hooksPath .githooks`
6. **Commit hooks to repository**
7. **Document in README**

---

**Last Updated**: October 24, 2025  
**Status**: Educational guide complete, ready for implementation

