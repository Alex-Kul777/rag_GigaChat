# Generate Command - Claude Rules

## Purpose
Generate new code or features for the RAG GigaChat project.

## Trigger
When user requests to generate new code, feature, or component.

## Process

### 1. Analysis Phase
- Understand the requirement
- Check existing codebase patterns
- Identify integration points
- Consider GigaChat API limitations

### 2. Generation Rules

#### New Component

[file: component_name.py]
"""
[component_name].py - [Brief description]

Added: [date]
Author: Claude
"""
[/file]

#### New Function

[python]
def function_name(param1: type, param2: type) -> return_type:
    """
    [Description in Russian/English]
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description
    
    Raises:
        ExceptionType: When something goes wrong
    """
    logger.info(f"Executing function_name with {param1}")
    # Implementation
    return result
[/python]

#### New Configuration

[python]
# In config.py
@dataclass
class NewConfig:
    """Configuration for new feature"""
    param1: str = "default"
    param2: int = 100
[/python]

### 3. Code Quality Requirements
- Add type hints
- Include docstrings
- Add logging at INFO level for major steps
- Add DEBUG logging for detailed steps
- Handle exceptions properly
- Follow existing naming conventions

### 4. Testing Requirements
- Provide test example in `if __name__ == "__main__"`
- Test with sample data
- Verify token limits
- Check performance

### 5. Documentation
- Update README.md if needed
- Add comments for complex logic
- Document new configuration parameters

## Output Format

### Generated: [Component Name]

#### Files Modified
- file1.py - Added function X
- config.py - Added parameter Y

#### Code Changes

[python]
# Code block with changes
[/python]

#### Usage Example

[python]
# How to use the new feature
[/python]

#### Testing

[bash]
# Commands to test
[/bash]
