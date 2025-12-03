# Nilai Python SDK

A Python SDK for the Nilai platform that provides delegation token management and OpenAI-compatible client functionality for accessing AI models through secure, decentralized infrastructure.

## 🚀 Quick Start

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Install with development dependencies
uv sync --group dev
```

### Basic Usage

```python
from nilai_py import Client

# Initialize client with API key
client = Client(
    base_url="https://testnet-p0.nilai.sandbox.nilogy.xyz/nuc/v1/",
    api_key="your-api-key-here"
)

# Make a chat completion request
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "user", "content": "Hello! Can you help me with something?"}
    ],
)

print(f"Response: {response.choices[0].message.content}")
```

## 📖 Usage Examples

### 1. API Key Mode (Simple)

The easiest way to get started. You'll need an API key from [nilpay.vercel.app](https://nilpay.vercel.app/).

```python
from nilai_py import Client

# Set up your API key in a .env file or environment variable
client = Client(
    base_url="https://testnet-p0.nilai.sandbox.nilogy.xyz/nuc/v1/",
    api_key="your-api-key-here"
)

# Make requests just like with OpenAI
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
)

print(response.choices[0].message.content)
```

### 2. Delegation Token Mode (Advanced)

For more secure, distributed access where you want to separate server credentials from client usage.

```python
from nilai_py import (
    Client,
    DelegationTokenServer,
    AuthType,
    DelegationServerConfig,
    NilAuthInstance
)

# Server-side: Create a delegation token server
server = DelegationTokenServer(
    private_key="your-private-key",
    config=DelegationServerConfig(
        nilauth_url=NilAuthInstance.SANDBOX.value,
        expiration_time=3600,  # 1 hour validity
        token_max_uses=10,     # Allow 10 uses
    )
)

# Client-side: Initialize client for delegation token mode
client = Client(
    base_url="https://nilai-a779.nillion.network/nuc/v1/",
    auth_type=AuthType.DELEGATION_TOKEN,
)

# Step 1: Client requests delegation
delegation_request = client.get_delegation_request()

# Step 2: Server creates delegation token
delegation_token = server.create_delegation_token(delegation_request)

# Step 3: Client uses the delegation token
client.update_delegation(delegation_token)

# Step 4: Make authenticated requests
response = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
    messages=[
        {"role": "user", "content": "What are the benefits of decentralized AI?"}
    ],
)

print(response.choices[0].message.content)
```

### 3. Environment Configuration

Create a `.env` file for your credentials:

```bash
# .env file
API_KEY=your-api-key-from-nilpay
PRIVATE_KEY=your-private-key-for-delegation-tokens
```

Then in your code:

```python
import os
from dotenv import load_dotenv
from nilai_py import Client

load_dotenv()

client = Client(
    base_url="https://testnet-p0.nilai.sandbox.nilogy.xyz/nuc/v1/",
    api_key=os.getenv("API_KEY")
)
```

## ✨ Features

- **🔐 Multiple Authentication Methods**: Support for API keys and delegation tokens
- **🤖 OpenAI Compatibility**: Drop-in replacement for OpenAI client in most cases
- **⚡ Automatic Token Management**: Handles token caching and expiration automatically
- **🛡️ Secure Delegation**: Server-side token management with configurable expiration and usage limits
- **🌐 Network Flexibility**: Support for sandbox and production environments
- **📝 Type Safety**: Full TypeScript-style type annotations for better IDE support

## 🏗️ Architecture

### DelegationTokenServer
Server-side component responsible for:
- Creating delegation tokens with configurable expiration and usage limits
- Managing root token lifecycle and caching
- Handling cryptographic operations securely

### Client
OpenAI-compatible client that:
- Supports both API key and delegation token authentication
- Automatically handles NUC token creation and management
- Provides familiar chat completion interface

### Token Management
- **Root Tokens**: Long-lived tokens for server authentication
- **Delegation Tokens**: Short-lived, limited-use tokens for client operations
- **Automatic Refresh**: Expired tokens are automatically refreshed when needed

## Features

- **DelegationTokenServer**: Server-side delegation token management
- **Client**: OpenAI-compatible client with Nilai authentication
- **Token Management**: Automatic token caching and expiration handling
- **Multiple Auth Methods**: Support for API keys and delegation tokens

## Testing

### Running Tests

To run all tests:
```bash
uv run pytest
```

To run tests for a specific module (e.g., server tests):
```bash
uv run pytest tests/test_server.py
```

To run tests with verbose output:
```bash
uv run pytest -v
```

To run tests for a specific test class:
```bash
uv run pytest tests/test_server.py::TestDelegationTokenServer -v
```

To run a specific test method:
```bash
uv run pytest tests/test_server.py::TestDelegationTokenServer::test_create_delegation_token_success -v
```

### Test Coverage

To run tests with coverage reporting:
```bash
uv run pytest --cov=nilai_py --cov-report=term-missing
```

To generate an HTML coverage report:
```bash
uv run pytest --cov=nilai_py --cov-report=html
```

### Current Test Coverage

The test suite provides comprehensive coverage:

| Module | Coverage | Details |
|--------|----------|---------|
| `src/nilai_py/server.py` | **100%** | Complete coverage of DelegationTokenServer class |
| `src/nilai_py/niltypes.py` | **100%** | Complete coverage of type definitions |
| `src/nilai_py/__init__.py` | **100%** | Module initialization |
| **Overall** | **71%** | High coverage across tested modules |

#### DelegationTokenServer Tests (16 test cases)

The `DelegationTokenServer` class has comprehensive test coverage including:

- ✅ **Initialization**: Default and custom configurations, invalid key handling
- ✅ **Token Expiration**: Expired/valid token detection, no expiration handling
- ✅ **Root Token Management**: Caching, automatic refresh, first access
- ✅ **Delegation Token Creation**: Success cases, configuration overrides, error handling
- ✅ **Error Handling**: Network failures, invalid cryptographic keys
- ✅ **Configuration**: Property access and instance management

### Test Structure

```
tests/
├── test_server.py          # DelegationTokenServer tests (100% coverage)
├── test_nilai_openai.py    # Client integration tests
└── config.py              # Test configuration
```

### Running Tests in Development

For continuous testing during development:
```bash
# Watch for file changes and rerun tests
uv run pytest --watch
```

### Test Dependencies

The following testing dependencies are included in the `dev` group:
- `pytest>=8.4.0`: Test framework
- `pytest-cov>=6.2.1`: Coverage reporting

## Development

### Code Quality

Run linting with:
```bash
uv run ruff check
uv run ruff format
```

### Adding New Tests

When adding new functionality:
1. Create corresponding test files in the `tests/` directory
2. Follow the existing naming convention (`test_*.py`)
3. Use descriptive test method names
4. Include docstrings explaining test purposes
5. Mock external dependencies appropriately
6. Aim for high test coverage

### Example Test Command Workflow

```bash
# 1. Install dependencies
uv sync --group dev

# 2. Run all tests with coverage
uv run pytest --cov=nilai_py --cov-report=term-missing

# 3. Run specific module tests
uv run pytest tests/test_server.py -v

# 4. Check code quality
uv run ruff check
uv run ruff format
```

## Project Structure

```
src/nilai_py/
├── __init__.py          # Package initialization
├── client.py           # OpenAI-compatible client
├── server.py           # DelegationTokenServer class
└── niltypes.py         # Type definitions
```
