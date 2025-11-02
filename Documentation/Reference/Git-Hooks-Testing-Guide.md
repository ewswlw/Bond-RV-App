# Git Hooks Testing Guide

**Created**: October 24, 2025  
**Last Updated**: October 24, 2025  
**Purpose**: Hands-on testing guide for Git hooks

---

## Quick Start

### 1. Setup Hooks (One-Time)

```bash
cd /path/to/Bond-RV-App
./setup_hooks.sh
```

### 2. Verify Setup

```bash
git config core.hooksPath
# Should output: .githooks
```

---

## Testing Each Hook

### Test 1: Pre-Commit Hook - Syntax Error Detection

**What it tests**: Python syntax validation

```bash
# Create a file with syntax error
echo "def broken_function(" > test_syntax_error.py

# Try to commit (should fail)
git add test_syntax_error.py
git commit -m "Test syntax error detection"
```

**Expected output**:
```
🔍 Running pre-commit checks...
🐍 Checking Python syntax...
❌ Syntax error in test_syntax_error.py
```

**Cleanup**:
```bash
rm test_syntax_error.py
```

---

### Test 2: Pre-Commit Hook - Large File Detection

**What it tests**: Prevents committing files > 10MB

```bash
# Create a large file (11MB)
dd if=/dev/zero of=large_file.bin bs=1M count=11

# Try to commit (should fail)
git add large_file.bin
git commit -m "Test large file detection"
```

**Expected output**:
```
📦 Checking for large files...
❌ ERROR: Large files detected (> 10MB):
11M     large_file.bin
```

**Cleanup**:
```bash
rm large_file.bin
```

---

### Test 3: Pre-Commit Hook - Sensitive Data Detection

**What it tests**: Detects passwords, API keys, secrets

```bash
# Create a file with sensitive data
echo "API_KEY = 'sk-1234567890abcdef'" > config_test.py

# Try to commit (should fail)
git add config_test.py
git commit -m "Test sensitive data detection"
```

**Expected output**:
```
🔒 Checking for sensitive data...
❌ ERROR: Potential sensitive data detected!
```

**Cleanup**:
```bash
rm config_test.py
```

---

### Test 4: Pre-Commit Hook - Debug Statement Warning

**What it tests**: Warns about print() statements

```bash
# Create a file with debug statements
cat > debug_test.py << 'EOF'
def process_data(data):
    print("Debug: processing data")  # Debug statement
    return data * 2
EOF

# Try to commit (should warn but allow)
git add debug_test.py
git commit -m "Test debug statement detection"
```

**Expected output**:
```
🐛 Checking for debug statements...
⚠️  WARNING: Debug statements found:
+    print("Debug: processing data")
```

**Note**: This is a warning, commit will proceed

**Cleanup**:
```bash
git reset HEAD~1
rm debug_test.py
```

---

### Test 5: Commit-Msg Hook - Short Message

**What it tests**: Minimum message length (10 chars)

```bash
# Try to commit with short message (should fail)
git commit --allow-empty -m "short"
```

**Expected output**:
```
📝 Validating commit message...
❌ ERROR: Commit message too short (minimum 10 characters)
   Your message: 'short'
   Length: 5 characters
```

---

### Test 6: Commit-Msg Hook - Valid Message

**What it tests**: Proper commit message format

```bash
# Commit with valid message (should succeed)
git commit --allow-empty -m "Add feature to pipeline module"
```

**Expected output**:
```
📝 Validating commit message...
ℹ️  Standard commit message (not using conventional commits)
✅ Commit message valid
```

**Cleanup**:
```bash
git reset HEAD~1
```

---

### Test 7: Commit-Msg Hook - Conventional Commits

**What it tests**: Conventional commits format recognition

```bash
# Commit with conventional format (should succeed)
git commit --allow-empty -m "feat(pipeline): add CUSIP validation"
```

**Expected output**:
```
📝 Validating commit message...
✅ Conventional commit format detected
   Type: feat
   Scope: pipeline
✅ Commit message valid
```

**Cleanup**:
```bash
git reset HEAD~1
```

---

### Test 8: Pre-Push Hook - Unit Tests

**What it tests**: Runs unit tests before push

**Setup**: Make sure tests exist and pass

```bash
# Run tests manually first
pytest tests/unit/ -v

# Make a commit
git commit --allow-empty -m "Test pre-push hook"

# Try to push (should run tests)
git push origin main
```

**Expected output**:
```
🚀 Running pre-push checks...
🧪 Running unit tests...
tests/unit/test_utils.py::TestDateParsing::test_parse_date_valid_formats PASSED
...
✅ All unit tests passed
```

**Note**: If you don't want to actually push, use `--dry-run`:
```bash
git push --dry-run origin main
```

---

### Test 9: Pre-Push Hook - Failing Tests

**What it tests**: Blocks push if tests fail

**Setup**: Temporarily break a test

```bash
# Edit a test to fail
cat >> tests/unit/test_utils.py << 'EOF'

def test_intentional_failure():
    """This test will fail"""
    assert False, "Intentional failure for testing"
EOF

# Commit the change
git add tests/unit/test_utils.py
git commit -m "Add failing test"

# Try to push (should fail)
git push origin main
```

**Expected output**:
```
🧪 Running unit tests...
FAILED tests/unit/test_utils.py::test_intentional_failure
❌ Unit tests failed!
Fix failing tests before pushing
```

**Cleanup**:
```bash
git reset HEAD~1
git checkout tests/unit/test_utils.py
```

---

### Test 10: Bypassing Hooks (Emergency)

**What it tests**: `--no-verify` flag

```bash
# Create a file with syntax error
echo "def broken(" > emergency_test.py

# Commit with --no-verify (should succeed)
git add emergency_test.py
git commit --no-verify -m "Emergency commit bypassing hooks"
```

**Expected output**: No hook output, commit succeeds

**Cleanup**:
```bash
git reset HEAD~1
rm emergency_test.py
```

**⚠️ WARNING**: Only use `--no-verify` in emergencies!

---

## Complete Test Suite

Run all tests in sequence:

```bash
#!/bin/bash
# Complete hook test suite

echo "🧪 Running complete Git hooks test suite..."
echo ""

# Test 1: Syntax error
echo "Test 1: Syntax error detection"
echo "def broken(" > test_syntax.py
git add test_syntax.py
if ! git commit -m "Test syntax" 2>&1 | grep -q "Syntax error"; then
    echo "❌ Test 1 failed"
else
    echo "✅ Test 1 passed"
fi
rm -f test_syntax.py
git reset HEAD 2>/dev/null || true
echo ""

# Test 2: Short commit message
echo "Test 2: Short commit message"
if ! git commit --allow-empty -m "short" 2>&1 | grep -q "too short"; then
    echo "❌ Test 2 failed"
else
    echo "✅ Test 2 passed"
fi
echo ""

# Test 3: Valid commit message
echo "Test 3: Valid commit message"
if git commit --allow-empty -m "Add feature to pipeline" 2>&1 | grep -q "Commit message valid"; then
    echo "✅ Test 3 passed"
    git reset HEAD~1
else
    echo "❌ Test 3 failed"
fi
echo ""

# Test 4: Conventional commit
echo "Test 4: Conventional commit format"
if git commit --allow-empty -m "feat(pipeline): add validation" 2>&1 | grep -q "Conventional commit"; then
    echo "✅ Test 4 passed"
    git reset HEAD~1
else
    echo "❌ Test 4 failed"
fi
echo ""

echo "🎉 Test suite complete!"
```

Save as `test_hooks.sh`, make executable, and run:

```bash
chmod +x test_hooks.sh
./test_hooks.sh
```

---

## Troubleshooting

### Hook Not Running

**Symptom**: Commit succeeds without any hook output

**Solutions**:

1. Check hooks path configuration:
   ```bash
   git config core.hooksPath
   ```

2. Re-run setup script:
   ```bash
   ./setup_hooks.sh
   ```

3. Check hook is executable:
   ```bash
   ls -la .githooks/
   ```

### Hook Fails Unexpectedly

**Symptom**: Hook fails but error message is unclear

**Solutions**:

1. Run hook manually:
   ```bash
   .githooks/pre-commit
   ```

2. Add debug output:
   ```bash
   # Add to top of hook script
   set -x  # Enable debug mode
   ```

3. Check exit code:
   ```bash
   .githooks/pre-commit
   echo $?  # 0 = success, non-zero = failure
   ```

### Permission Denied

**Symptom**: `Permission denied` when running hook

**Solution**:
```bash
chmod +x .githooks/*
```

### Hook Runs on Wrong Files

**Symptom**: Hook checks files not in current commit

**Solution**: Hooks should use `--cached` flag:
```bash
git diff --cached --name-only  # Only staged files
```

---

## Best Practices

### 1. Test Hooks Regularly

- Test after modifying hooks
- Test with various file types
- Test edge cases

### 2. Document Hook Behavior

- Update this guide when adding new checks
- Include examples of failures
- Explain how to fix common issues

### 3. Keep Hooks Fast

- Hooks should complete in < 5 seconds
- Use pre-push for slow checks
- Cache results when possible

### 4. Provide Clear Feedback

- Use emoji and colors
- Explain what failed
- Show how to fix issues

### 5. Allow Escape Hatch

- Always allow `--no-verify`
- Document when to use it
- Use warnings for minor issues

---

## Advanced Testing

### Test Hook Performance

```bash
time .githooks/pre-commit
```

**Target**: < 5 seconds

### Test with Multiple Files

```bash
# Create multiple test files
for i in {1..10}; do
    echo "def func$i(): pass" > test_file_$i.py
done

# Commit all at once
git add test_file_*.py
git commit -m "Test multiple files"

# Cleanup
rm test_file_*.py
git reset HEAD~1
```

### Test Concurrent Commits

```bash
# Simulate multiple developers
git checkout -b feature1
echo "# Feature 1" > feature1.py
git add feature1.py
git commit -m "Add feature 1"

git checkout main
git checkout -b feature2
echo "# Feature 2" > feature2.py
git add feature2.py
git commit -m "Add feature 2"

# Cleanup
git checkout main
git branch -D feature1 feature2
```

---

## Summary

### Essential Tests:
1. ✅ Syntax error detection
2. ✅ Large file detection
3. ✅ Sensitive data detection
4. ✅ Commit message validation
5. ✅ Unit tests before push

### Optional Tests:
6. ⏳ Debug statement warnings
7. ⏳ Code formatting checks
8. ⏳ Documentation timestamps
9. ⏳ Merge conflict detection

### Emergency Procedures:
- Use `--no-verify` only when necessary
- Document why you bypassed hooks
- Fix issues in next commit

---

**Last Updated**: October 24, 2025  
**Status**: Complete testing guide with 10+ test scenarios

