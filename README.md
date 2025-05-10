
# AutoGenLib

> The only library you'll need ever.
>
> Import wisdom, export code.

AutoGenLib is a Python library that automatically generates code on-the-fly using OpenAI's API. When you try to import a module or function that doesn't exist, AutoGenLib creates it for you based on a high-level description of what you need.

## Features

- **Dynamic Code Generation**: Import modules and functions that don't exist yet
- **Context-Aware**: New functions are generated with knowledge of existing code
- **Progressive Enhancement**: Add new functionality to existing modules seamlessly
- **Persistent Storage**: Generated code is cached for future use
- **Full Codebase Context**: LLM can see all previously generated modules for better consistency

## Installation

```bash
pip install autogenlib
```

Or install from source:

```bash
git clone https://github.com/cofob/autogenlib.git
cd autogenlib
pip install -e .
```

## Requirements

- Python 3.12+
- OpenAI API key

## Quick Start

Set OpenAI API key in `OPENAI_API_KEY` env variable.

```python
from autogenlib import init
init("Library for generating secure random tokens")

# Import a function that doesn't exist yet - it will be automatically generated
from autogenlib.tokens import generate_token

# Use the generated function
token = generate_token(length=32)
print(token)
```

## How It Works

1. You initialize AutoGenLib with a description of what you need
2. When you import a module or function under the `autogenlib` namespace, the library:
   - Checks if the module/function already exists
   - If not, it sends a request to OpenAI's API with your description
   - The API generates the appropriate code
   - The code is validated, cached, and executed
   - The requested module/function becomes available for use

## Examples

### Generate a TOTP Generator

```python
from autogenlib import init
init("Library for generating TOTP codes")
from autogenlib.totp import totp_generator
print(totp_generator("SECRETKEY123"))
```

### Add a Verification Function Later

```python
# Later in your application, you need verification:
from autogenlib.totp import verify_totp
result = verify_totp("SECRETKEY123", "123456")
print(f"Verification result: {result}")
```

### Create Multiple Modules

```python
from autogenlib import init
init("Cryptographic utility library")

# Generate encryption module
from autogenlib.encryption import encrypt_text, decrypt_text
encrypted = encrypt_text("Secret message", "password123")
decrypted = decrypt_text(encrypted, "password123")
print(decrypted)

# Generate hashing module
from autogenlib.hashing import hash_password, verify_password
hashed = hash_password("my_secure_password")
is_valid = verify_password("my_secure_password", hashed)
print(f"Password valid: {is_valid}")
```

## Configuration

### Setting the OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
# Optional
export OPENAI_API_BASE_URL="https://openrouter.ai/api/v1"  # Use OpenRouter API
export OPENAI_API_KEY="openai/gpt-4.1"
```

Or in your Python code (not recommended for production):

```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Cache Location

By default, generated code is cached in `~/.autogenlib_cache`. You can check this directory to see what has been generated.

## Limitations

- Requires internet connection to generate new code
- Depends on OpenAI API availability
- Generated code quality depends on the clarity of your description
- Not suitable for production-critical code without review

## Advanced Usage

### Inspecting Generated Code

You can inspect the code that was generated for a module:

```python
from autogenlib.totp import totp_generator
import inspect
print(inspect.getsource(totp_generator))
```

## How AutoGenLib Uses the OpenAI API

AutoGenLib creates prompts for the OpenAI API that include:

1. The description you provided
2. Any existing code in the module being enhanced
3. The full context of all previously generated modules
4. The specific function or feature needed

This comprehensive context helps the LLM generate code that's consistent with your existing codebase.

## Contributing

Contributions are not welcome! This is just a fun PoC project.

## License

MIT License

---

*Note: This library is meant for prototyping and experimentation. Always review automatically generated code before using it in production environments.*
*Note: Of course 100% of the code of this library was generated via LLM*
