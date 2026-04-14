# Improve Command - Claude Rules

## Purpose
Improve existing code quality, performance, or readability.

## Trigger
When user requests to improve, refactor, or optimize existing code.

## Process

### 1. Analysis Phase
- Identify code to improve
- Check for:
  - Performance bottlenecks
  - Code duplication
  - Missing error handling
  - Inefficient algorithms
  - Memory usage issues
  - Token limit compliance

### 2. Improvement Areas

#### Performance
- Reduce API calls (batch embeddings)
- Optimize FAISS search parameters
- Cache expensive operations
- Reduce token usage for GigaChat

#### Code Quality
- Extract repeated logic into functions
- Simplify complex conditions
- Add missing type hints
- Improve variable naming

#### Error Handling
- Add specific exception types
- Implement retry logic for API calls
- Add graceful degradation
- Log errors with context

#### Token Optimization
- Reduce chunk_size for large documents
- Implement text truncation for long contexts
- Batch embedding requests
- Add token counting before API calls

### 3. Implementation Rules

#### Before/After Format

[section: BEFORE]
[python]
# Original code
[/python]

[section: AFTER]
[python]
# Improved code
[/python]

[section: CHANGES]
- Change 1: Description
- Change 2: Description

[section: IMPACT]
- Performance: +/- X%
- Memory: +/- X MB
- Token usage: +/- X tokens

### 4. Validation
- Test with same inputs
- Verify outputs unchanged
- Check error handling
- Measure improvements

### 5. Documentation
- Update docstrings
- Add comments for complex improvements
- Document breaking changes

## Common Improvements Patterns

### Pattern 1: Batch Processing
[python]
# Before: Single request
for doc in documents:
    embedding = api.embed(doc)

# After: Batch requests
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    embeddings = api.embed_batch(batch)
[/python]

### Pattern 2: Caching
[python]
# Before: Repeated computation
result = expensive_function(data)

# After: With caching
@lru_cache(maxsize=128)
def cached_function(data):
    return expensive_function(data)
[/python]

### Pattern 3: Error Handling
[python]
# Before: No error handling
result = api.call()

# After: With retry logic
for attempt in range(max_retries):
    try:
        result = api.call()
        break
    except APIError as e:
        if attempt == max_retries - 1:
            raise
        time.sleep(2 ** attempt)
[/python]

## Code Review Checklist

When improving code, verify:
- [ ] No breaking changes unless documented
- [ ] Tests pass after changes
- [ ] Performance improved or not degraded
- [ ] Memory usage acceptable
- [ ] Error handling complete
- [ ] Logging appropriate
- [ ] Documentation updated

## Output Format

[header: IMPROVEMENT SUMMARY]
File: path/to/file.py
Function: function_name
Type: Performance / Quality / Error Handling / Token Optimization

[header: BEFORE]
[python]
Original code
[/python]

[header: AFTER]
[python]
Improved code
[/python]

[header: CHANGES MADE]
1. Change description
2. Change description

[header: IMPACT]
- Lines of code: +X -Y
- Performance: description
- Memory: description

[header: TESTING]
[python]
# Test code
[/python]
