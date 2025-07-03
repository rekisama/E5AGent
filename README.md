# AutoGen Self-Expanding Agent System

A secure, self-improving AI agent system with enhanced function management capabilities.

## Core Features

- **Secure Function Registry**: Thread-safe function storage with comprehensive metadata
- **Enhanced Function Tools**: Secure code execution with AST-based validation
- **Cross-Platform Security**: Multiprocessing-based timeout and sandboxed execution
- **Intelligent Test Generation**: Automatic test case generation with pattern recognition
- **Flexible Result Comparison**: Float tolerance and deep structure comparison

## Project Structure

```
autogen_godel_agent/
├── agents/                    # Agent implementations
│   ├── function_creator_agent.py
│   ├── planner_agent.py
│   └── test_case_generator.py
├── tools/                     # Core tools and utilities
│   ├── function_registry.py   # Enhanced function registry
│   ├── function_tools.py      # Enhanced function tools
│   ├── secure_executor.py     # Secure code execution
│   └── test_runner.py         # Enhanced test runner
├── memory/                    # Persistent storage
├── config.py                  # Configuration settings
├── demo.py                    # Demonstration script
├── interactive.py             # Interactive interface
├── main.py                    # Main entry point
└── requirements.txt           # Dependencies
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r autogen_godel_agent/requirements.txt
   ```

2. Run the demo:
   ```bash
   python autogen_godel_agent/demo.py
   ```

3. Start interactive mode:
   ```bash
   python autogen_godel_agent/interactive.py
   ```

## Security Features

- AST-based code validation
- Sandboxed execution environment
- Cross-platform timeout protection
- Restricted built-in functions
- Safe module import whitelist
- Comprehensive error handling

## Core Components

### Function Registry
- Thread-safe function storage
- Comprehensive metadata management
- Version control and usage tracking
- Cross-platform file locking

### Function Tools
- Secure code validation
- Enhanced test generation
- Pre-registration testing
- Flexible result comparison

### Secure Executor
- AST security scanning
- Sandboxed code execution
- Timeout protection
- Safe module imports

### Test Runner
- Cross-platform timeout using multiprocessing
- Flexible result comparison (float tolerance, deep comparison)
- Support for class methods and complex expressions
- Comprehensive error handling

## Usage Example

```python
from autogen_godel_agent.tools.function_tools import get_function_tools

# Get function tools instance
tools = get_function_tools()

# Register a function with automatic testing
success = tools.register_function(
    func_name="calculate_area",
    func_code='''
def calculate_area(radius: float) -> float:
    """Calculate circle area."""
    import math
    return math.pi * radius * radius
''',
    description="Calculate circle area",
    require_tests=True,
    auto_generate_tests=True
)

# Search and use functions
functions = tools.search_functions("area")
info = tools.get_function_info("calculate_area")
```

## Configuration

Edit `autogen_godel_agent/config.py` to customize:
- API keys and endpoints
- Security settings
- Timeout values
- Registry paths
