# Claude Slash Commands Guide for Bond Pipeline Project

**Created**: October 30, 2025  
**Last Updated**: October 30, 2025  
**Purpose**: Complete guide to using Claude's slash commands for development

---

## Table of Contents

1. [What Are Slash Commands?](#what-are-slash-commands)
2. [Available Slash Commands](#available-slash-commands)
3. [Practical Examples for Bond Pipeline](#practical-examples-for-bond-pipeline)
4. [Advanced Workflows](#advanced-workflows)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## What Are Slash Commands?

**Slash commands** are special commands in Claude that start with `/` and provide quick access to common development tasks. They're like shortcuts that help you work more efficiently with your codebase.

### Key Benefits:

- **Speed**: Quick access to common tasks
- **Context**: Claude understands your project structure
- **Intelligence**: Smart suggestions based on your code
- **Integration**: Works seamlessly with your files

---

## Available Slash Commands

### 1. `/file` - Work with Files

**Purpose**: Reference, read, or modify specific files

**Syntax**:
```
/file path/to/file.py
```

**Use Cases**:
- View file contents
- Analyze code structure
- Get suggestions for improvements
- Debug specific files

**Example**:
```
/file bond_pipeline/utils.py
Can you explain the CUSIP validation logic?
```

---

### 2. `/project` - Project-Wide Operations

**Purpose**: Understand and work with entire project

**Syntax**:
```
/project
```

**Use Cases**:
- Get project overview
- Understand architecture
- Find files by functionality
- Identify dependencies

**Example**:
```
/project
Show me all files related to data transformation
```

---

### 3. `/search` - Search Codebase

**Purpose**: Find specific code, functions, or patterns

**Syntax**:
```
/search pattern
```

**Use Cases**:
- Find function definitions
- Locate where variables are used
- Search for TODO comments
- Find similar code patterns

**Example**:
```
/search validate_cusip
Where is this function used?
```

---

### 4. `/explain` - Code Explanation

**Purpose**: Get detailed explanations of code

**Syntax**:
```
/explain file.py:function_name
```

**Use Cases**:
- Understand complex logic
- Learn how functions work
- Get documentation suggestions
- Identify potential issues

**Example**:
```
/explain bond_pipeline/transform.py:deduplicate_cusips
```

---

### 5. `/fix` - Debug and Fix Issues

**Purpose**: Identify and fix bugs

**Syntax**:
```
/fix file.py
```

**Use Cases**:
- Debug errors
- Fix syntax issues
- Improve code quality
- Optimize performance

**Example**:
```
/fix bond_pipeline/extract.py
The date parsing is failing for some files
```

---

### 6. `/test` - Generate Tests

**Purpose**: Create unit tests for code

**Syntax**:
```
/test file.py
```

**Use Cases**:
- Generate test cases
- Improve test coverage
- Create edge case tests
- Mock external dependencies

**Example**:
```
/test bond_pipeline/load.py
Generate tests for parquet writing functions
```

---

### 7. `/refactor` - Code Refactoring

**Purpose**: Improve code structure and quality

**Syntax**:
```
/refactor file.py
```

**Use Cases**:
- Simplify complex code
- Extract functions
- Improve naming
- Apply design patterns

**Example**:
```
/refactor bond_pipeline/pipeline.py
Make the main function more modular
```

---

### 8. `/docs` - Generate Documentation

**Purpose**: Create or improve documentation

**Syntax**:
```
/docs file.py
```

**Use Cases**:
- Generate docstrings
- Create README files
- Write API documentation
- Add inline comments

**Example**:
```
/docs bond_pipeline/utils.py
Add comprehensive docstrings to all functions
```

---

## Practical Examples for Bond Pipeline

### Example 1: Understanding the Pipeline

**Goal**: Understand how the pipeline processes data

**Commands**:
```
/project
Show me the data flow from Excel files to Parquet output

/file bond_pipeline/pipeline.py
Explain the main orchestration logic

/search process_files
Show me all functions involved in file processing
```

**Result**: Complete understanding of data flow

---

### Example 2: Adding New Feature

**Goal**: Add ISIN validation alongside CUSIP validation

**Commands**:
```
/file bond_pipeline/utils.py
Show me the CUSIP validation function

/explain bond_pipeline/utils.py:validate_cusip
How does this validation work?

# After understanding:
Can you create a similar validate_isin function?

/test bond_pipeline/utils.py
Generate tests for the new validate_isin function
```

**Result**: New feature with tests

---

### Example 3: Debugging Issue

**Goal**: Fix date parsing errors

**Commands**:
```
/file bond_pipeline/extract.py
I'm getting errors parsing dates from filenames

/fix bond_pipeline/extract.py
The parse_date_from_filename function fails for "API 02.29.23.xlsx"

/search parse_date
Show me all places where dates are parsed

/test bond_pipeline/extract.py
Add test cases for leap year dates
```

**Result**: Bug fixed with new tests

---

### Example 4: Improving Code Quality

**Goal**: Refactor transform module

**Commands**:
```
/file bond_pipeline/transform.py
This file is getting too long

/refactor bond_pipeline/transform.py
Split into smaller, focused functions

/docs bond_pipeline/transform.py
Add docstrings explaining each transformation step

/test bond_pipeline/transform.py
Ensure all refactored functions have tests
```

**Result**: Cleaner, well-documented code

---

### Example 5: Adding Test Coverage

**Goal**: Increase test coverage to 95%

**Commands**:
```
/project
Show me which files have low test coverage

/file tests/unit/test_utils.py
What test cases are we missing?

/test bond_pipeline/extract.py
Generate comprehensive tests for all edge cases

/test bond_pipeline/load.py
Add tests for append vs override modes
```

**Result**: Improved test coverage

---

### Example 6: Performance Optimization

**Goal**: Speed up pipeline execution

**Commands**:
```
/file bond_pipeline/pipeline.py
Can you identify performance bottlenecks?

/explain bond_pipeline/load.py:save_to_parquet
Is this function optimized?

/refactor bond_pipeline/extract.py
Can we read Excel files in parallel?

/test bond_pipeline/pipeline.py
Add performance benchmarking tests
```

**Result**: Faster pipeline

---

### Example 7: Adding New Data Source

**Goal**: Support CSV files in addition to Excel

**Commands**:
```
/file bond_pipeline/extract.py
Show me the Excel reading logic

Can you add support for CSV files with similar structure?

/test bond_pipeline/extract.py
Generate tests for CSV file reading

/docs bond_pipeline/extract.py
Document the new CSV support
```

**Result**: Multi-format support

---

### Example 8: Creating API Endpoint

**Goal**: Add REST API to query parquet data

**Commands**:
```
/project
What's the best way to add a REST API to this project?

Create a new file api/endpoints.py with FastAPI endpoints for:
- GET /bonds - List all bonds
- GET /bonds/{cusip} - Get bond by CUSIP
- GET /universe - Get current universe

/test api/endpoints.py
Generate integration tests for all endpoints

/docs api/endpoints.py
Create API documentation with examples
```

**Result**: REST API with documentation

---

## Advanced Workflows

### Workflow 1: Feature Development

**Step-by-step process**:

```
1. /project
   "I want to add email notifications when pipeline completes"

2. /file bond_pipeline/config.py
   "Where should I add email configuration?"

3. Create new file utils/email_notifier.py
   "Can you create an email notification module?"

4. /test utils/email_notifier.py
   "Generate tests with mocked SMTP"

5. /file bond_pipeline/pipeline.py
   "Integrate email notifications at the end"

6. /docs utils/email_notifier.py
   "Add comprehensive documentation"
```

---

### Workflow 2: Bug Investigation

**Step-by-step process**:

```
1. /search "duplicate"
   "Show me all duplicate handling code"

2. /file bond_pipeline/transform.py
   "The deduplicate function isn't working correctly"

3. /explain bond_pipeline/transform.py:deduplicate_cusips
   "Walk me through the logic"

4. /fix bond_pipeline/transform.py
   "It's keeping first instead of last occurrence"

5. /test bond_pipeline/transform.py
   "Add test to verify last occurrence is kept"

6. /file tests/unit/test_transform.py
   "Run the new test to confirm fix"
```

---

### Workflow 3: Code Review

**Step-by-step process**:

```
1. /project
   "Review the entire codebase for improvements"

2. /file bond_pipeline/utils.py
   "Are there any code quality issues?"

3. /refactor bond_pipeline/utils.py
   "Apply suggested improvements"

4. /docs bond_pipeline/utils.py
   "Improve documentation"

5. /test bond_pipeline/utils.py
   "Ensure tests still pass after refactoring"
```

---

### Workflow 4: Documentation Sprint

**Step-by-step process**:

```
1. /project
   "List all files missing documentation"

2. /docs bond_pipeline/config.py
   "Add module and class docstrings"

3. /docs bond_pipeline/utils.py
   "Add function docstrings with examples"

4. /docs bond_pipeline/extract.py
   "Document the extraction process"

5. Create Documentation/API/api-reference.md
   "Generate complete API reference"
```

---

## Best Practices

### 1. **Start with Context**

**Good**:
```
/project
I want to understand the data validation process before making changes
```

**Why**: Claude gets full context of your project

---

### 2. **Be Specific**

**Good**:
```
/file bond_pipeline/utils.py
Explain the validate_cusip function and how it handles edge cases
```

**Bad**:
```
/file bond_pipeline/utils.py
Explain everything
```

**Why**: Specific questions get specific answers

---

### 3. **Chain Commands**

**Good**:
```
/file bond_pipeline/extract.py
Show me the date parsing logic

/explain bond_pipeline/extract.py:parse_date_from_filename
How does this handle leap years?

/test bond_pipeline/extract.py
Add test cases for leap year dates
```

**Why**: Build understanding progressively

---

### 4. **Use for Learning**

**Good**:
```
/explain bond_pipeline/transform.py
I'm new to pandas, can you explain the deduplication logic?
```

**Why**: Claude can teach you while you work

---

### 5. **Combine with Regular Questions**

**Good**:
```
/file bond_pipeline/pipeline.py

I see you're using argparse. Would click be better for this use case?
Can you show me how to refactor to click?
```

**Why**: Get both code and advice

---

### 6. **Iterate and Refine**

**Good**:
```
/refactor bond_pipeline/utils.py
Make validate_cusip more modular

# After seeing result:
Can you also add type hints and improve error messages?

# After that:
Now add comprehensive docstrings
```

**Why**: Incremental improvements

---

## Real-World Scenarios

### Scenario 1: New Team Member Onboarding

**Goal**: Understand the codebase quickly

```
Day 1:
/project
Give me a high-level overview of this bond pipeline project

/file README.md
What are the main features?

/file bond_pipeline/pipeline.py
Show me the entry point

Day 2:
/explain bond_pipeline/extract.py
How does data extraction work?

/explain bond_pipeline/transform.py
What transformations are applied?

Day 3:
/test bond_pipeline/utils.py
Show me example test cases to understand expected behavior
```

---

### Scenario 2: Production Bug

**Goal**: Fix critical bug quickly

```
/search "Date"
Find all date-related code

/file bond_pipeline/extract.py
The pipeline is failing on files from 2024

/fix bond_pipeline/extract.py
Error: "ValueError: day is out of range for month"

/test bond_pipeline/extract.py
Add test case that reproduces the bug

# After fix:
/test bond_pipeline/extract.py
Verify the fix with comprehensive date tests
```

---

### Scenario 3: Feature Request

**Goal**: Add new column to universe table

```
/file bond_pipeline/config.py
Show me the UNIVERSE_COLUMNS definition

/file bond_pipeline/load.py
How is the universe table created?

Can you add "Maturity_Years" column to universe?
It should calculate years until maturity from Worst Date

/test bond_pipeline/load.py
Generate tests for the new column calculation

/docs bond_pipeline/load.py
Document the new Maturity_Years column
```

---

### Scenario 4: Code Audit

**Goal**: Security and quality review

```
/project
Identify potential security issues

/search "password|secret|key"
Find any hardcoded credentials

/file bond_pipeline/utils.py
Review for SQL injection vulnerabilities

/file bond_pipeline/extract.py
Check for path traversal issues

/refactor bond_pipeline/config.py
Move hardcoded values to environment variables
```

---

## Command Combinations

### Combination 1: Full Feature Implementation

```
/project → /file → /explain → Code → /test → /docs → /refactor
```

**Example**:
```
/project - Understand where feature fits
/file - Review related code
/explain - Understand existing logic
[Implement feature]
/test - Generate tests
/docs - Add documentation
/refactor - Clean up if needed
```

---

### Combination 2: Bug Fix Workflow

```
/search → /file → /fix → /test → /docs
```

**Example**:
```
/search - Find related code
/file - Review the buggy file
/fix - Get fix suggestions
/test - Add regression tests
/docs - Update documentation
```

---

### Combination 3: Refactoring Sprint

```
/file → /refactor → /test → /docs → /explain
```

**Example**:
```
/file - Review current code
/refactor - Get refactoring suggestions
/test - Ensure tests still pass
/docs - Update documentation
/explain - Verify improvements
```

---

## Tips and Tricks

### Tip 1: Use with Git Diff

```
# After making changes
git diff bond_pipeline/utils.py

/file bond_pipeline/utils.py
Review my changes and suggest improvements
```

---

### Tip 2: Batch Operations

```
/file bond_pipeline/utils.py
/file bond_pipeline/extract.py
/file bond_pipeline/transform.py

Review all three files for consistency in error handling
```

---

### Tip 3: Context Building

```
/project
/file bond_pipeline/config.py
/file bond_pipeline/utils.py

Now that you understand the structure, help me add logging throughout
```

---

### Tip 4: Learning Mode

```
/explain bond_pipeline/transform.py:deduplicate_cusips

Can you explain this line by line?
What pandas methods are being used?
Are there more efficient alternatives?
```

---

### Tip 5: Pair Programming

```
/file bond_pipeline/pipeline.py

I want to add progress bars. Can you:
1. Suggest the best library (tqdm?)
2. Show me where to add progress tracking
3. Generate example code
4. Create tests
```

---

## Troubleshooting

### Issue 1: Command Not Working

**Problem**: Slash command doesn't seem to work

**Solutions**:
1. Make sure command starts with `/`
2. Check file path is correct
3. Verify file exists in project
4. Try `/project` first for context

---

### Issue 2: Unclear Results

**Problem**: Claude's response isn't what you expected

**Solutions**:
1. Be more specific in your question
2. Provide more context
3. Use `/explain` to clarify
4. Break down into smaller questions

---

### Issue 3: File Not Found

**Problem**: `/file path/to/file.py` returns error

**Solutions**:
1. Check file path is relative to project root
2. Use `/project` to see available files
3. Verify file name spelling
4. Check file extension

---

## Summary

### Essential Commands for Bond Pipeline:

| Task | Command | Example |
|------|---------|---------|
| **Understand code** | `/explain` | `/explain bond_pipeline/utils.py:validate_cusip` |
| **Fix bugs** | `/fix` | `/fix bond_pipeline/extract.py` |
| **Add tests** | `/test` | `/test bond_pipeline/load.py` |
| **Refactor** | `/refactor` | `/refactor bond_pipeline/pipeline.py` |
| **Document** | `/docs` | `/docs bond_pipeline/utils.py` |
| **Search** | `/search` | `/search validate_cusip` |
| **Overview** | `/project` | `/project` |
| **View file** | `/file` | `/file bond_pipeline/config.py` |

### Recommended Workflow:

1. **Start**: `/project` - Get context
2. **Explore**: `/file` - View specific files
3. **Understand**: `/explain` - Learn the code
4. **Modify**: Make changes
5. **Test**: `/test` - Generate tests
6. **Document**: `/docs` - Add documentation
7. **Refine**: `/refactor` - Improve quality

---

## Next Steps

1. **Try basic commands**: Start with `/project` and `/file`
2. **Practice workflows**: Follow the examples above
3. **Experiment**: Combine commands creatively
4. **Build habits**: Use commands daily
5. **Share**: Teach team members

---

**Last Updated**: October 30, 2025  
**Status**: Complete guide with practical examples

