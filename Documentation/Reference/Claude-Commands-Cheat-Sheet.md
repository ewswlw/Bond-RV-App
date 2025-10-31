# Claude Slash Commands Cheat Sheet

**Created**: October 30, 2025  
**Last Updated**: October 30, 2025  
**Purpose**: Quick reference for Claude slash commands

---

## Quick Reference

### Basic Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `/project` | Project overview | `/project` |
| `/file <path>` | View/edit file | `/file bond_pipeline/utils.py` |
| `/search <term>` | Search codebase | `/search validate_cusip` |
| `/explain <path>` | Explain code | `/explain bond_pipeline/utils.py:validate_cusip` |
| `/fix <path>` | Debug/fix issues | `/fix bond_pipeline/extract.py` |
| `/test <path>` | Generate tests | `/test bond_pipeline/load.py` |
| `/refactor <path>` | Improve code | `/refactor bond_pipeline/pipeline.py` |
| `/docs <path>` | Add documentation | `/docs bond_pipeline/utils.py` |

---

## Common Workflows

### 1. Understanding New Code

```
/project
/file bond_pipeline/pipeline.py
/explain bond_pipeline/pipeline.py:main
```

### 2. Fixing a Bug

```
/search <error_message>
/file <buggy_file.py>
/fix <buggy_file.py>
/test <buggy_file.py>
```

### 3. Adding a Feature

```
/project
/file <related_file.py>
[implement feature]
/test <new_file.py>
/docs <new_file.py>
```

### 4. Refactoring

```
/file <messy_file.py>
/refactor <messy_file.py>
/test <messy_file.py>
/docs <messy_file.py>
```

### 5. Increasing Test Coverage

```
/project
/test bond_pipeline/extract.py
/test bond_pipeline/transform.py
/test bond_pipeline/load.py
```

---

## Bond Pipeline Specific Examples

### Understand CUSIP Validation

```
/file bond_pipeline/utils.py
/explain bond_pipeline/utils.py:validate_cusip
Show me examples of valid and invalid CUSIPs
```

### Debug Date Parsing

```
/file bond_pipeline/extract.py
/fix bond_pipeline/extract.py
The date parsing fails for leap year dates
```

### Add New Column to Universe

```
/file bond_pipeline/config.py
/file bond_pipeline/load.py
Add "Maturity_Years" column to UNIVERSE_COLUMNS
```

### Optimize Performance

```
/file bond_pipeline/pipeline.py
Identify performance bottlenecks
Can we parallelize Excel file reading?
```

### Generate Missing Tests

```
/test bond_pipeline/extract.py
Focus on edge cases: leap years, invalid dates, missing files
```

---

## Pro Tips

### 1. Chain Commands for Context

```
/project
/file bond_pipeline/utils.py
/explain bond_pipeline/utils.py:validate_cusip
Now help me add ISIN validation
```

### 2. Be Specific

**Good**: `/file bond_pipeline/utils.py - Explain the CUSIP validation logic`  
**Bad**: `/file bond_pipeline/utils.py - Explain everything`

### 3. Use for Learning

```
/explain bond_pipeline/transform.py
I'm new to pandas, explain the deduplication logic step by step
```

### 4. Combine with Questions

```
/file bond_pipeline/pipeline.py
Should we use click instead of argparse? Show me how to refactor
```

### 5. Iterate

```
/refactor bond_pipeline/utils.py
[review suggestions]
Also add type hints and improve error messages
[review again]
Now add comprehensive docstrings
```

---

## Common Patterns

### Pattern 1: Full Feature Development

```
1. /project - Understand context
2. /file - Review related code
3. /explain - Understand logic
4. [Implement feature]
5. /test - Generate tests
6. /docs - Add documentation
7. /refactor - Clean up
```

### Pattern 2: Bug Investigation

```
1. /search - Find related code
2. /file - View buggy file
3. /explain - Understand logic
4. /fix - Get fix suggestions
5. /test - Add regression tests
```

### Pattern 3: Code Review

```
1. /project - Get overview
2. /file - Review each file
3. /refactor - Get improvement suggestions
4. /test - Ensure tests pass
5. /docs - Update documentation
```

---

## Keyboard Shortcuts

- Type `/` to see available commands
- Press `Tab` to autocomplete file paths
- Use `↑` `↓` arrows to navigate command history

---

## When to Use Each Command

### Use `/project` when:
- Starting new task
- Need project overview
- Want to find files
- Understanding architecture

### Use `/file` when:
- Need to view specific file
- Want to modify code
- Analyzing file structure
- Reviewing changes

### Use `/search` when:
- Looking for specific function
- Finding where code is used
- Searching for patterns
- Locating TODOs

### Use `/explain` when:
- Don't understand code
- Learning new concepts
- Reviewing complex logic
- Need documentation

### Use `/fix` when:
- Debugging errors
- Code not working
- Need bug suggestions
- Optimizing performance

### Use `/test` when:
- Need test cases
- Improving coverage
- Testing edge cases
- Mocking dependencies

### Use `/refactor` when:
- Code is messy
- Want better structure
- Improving readability
- Applying patterns

### Use `/docs` when:
- Missing documentation
- Need docstrings
- Creating README
- API documentation

---

## Troubleshooting

### Command doesn't work
- Check spelling
- Verify file path
- Use `/project` first
- Be more specific

### Unclear results
- Add more context
- Be more specific
- Break into smaller questions
- Use `/explain` to clarify

### File not found
- Check path from project root
- Use `/project` to see files
- Verify file name
- Check extension

---

## Quick Wins

### 1. Instant Documentation
```
/docs bond_pipeline/utils.py
Add comprehensive docstrings to all functions
```

### 2. Test Generation
```
/test bond_pipeline/extract.py
Generate tests for all edge cases
```

### 3. Code Explanation
```
/explain bond_pipeline/transform.py:deduplicate_cusips
Explain this function line by line
```

### 4. Bug Fixes
```
/fix bond_pipeline/pipeline.py
The pipeline crashes on empty Excel files
```

### 5. Refactoring
```
/refactor bond_pipeline/load.py
Make this more modular and add error handling
```

---

## Remember

- **Start with `/project`** for context
- **Be specific** in your questions
- **Chain commands** for better results
- **Iterate** to refine output
- **Combine** with regular questions

---

**Print this out and keep it handy!** 📄

**Last Updated**: October 30, 2025

