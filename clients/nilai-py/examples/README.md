# Nilai Python SDK Examples

This directory contains example scripts demonstrating how to use the Nilai Python SDK.

## Examples

### 1. API Key Mode (`0-api_key_mode.py`)
Basic example showing how to use the SDK with an API key for authentication.

### 2. Delegation Token Mode (`1-delegation_token_mode.py`)
Advanced example showing how to use delegation tokens for authentication, including server-side token generation.

### 3. Streaming Mode (`2-streaming_mode.py`)
Basic streaming example showing how to receive real-time responses from the API.

### 4. Advanced Streaming (`3-advanced_streaming.py`)
Advanced streaming example with error handling, progress tracking, and custom processing.

## Configuration

All examples use the `config.py` file to load the API key from environment variables. Make sure to:

1. Create a `.env` file in the project root
2. Add your API key: `API_KEY=your_api_key_here`
3. Or set the environment variable directly: `export API_KEY=your_api_key_here`

## Running Examples

```bash
# Basic API key example
python examples/0-api_key_mode.py

# Delegation token example
python examples/1-delegation_token_mode.py

# Streaming example
python examples/2-streaming_mode.py

# Advanced streaming example
python examples/3-advanced_streaming.py
```

## Streaming Features

The streaming examples demonstrate:

- **Real-time response processing**: Receive and display responses as they're generated
- **Progress tracking**: Monitor chunk count and response length
- **Error handling**: Graceful handling of interruptions and errors
- **Custom processing**: Word counting, line tracking, and other real-time analysis
- **Retry logic**: Automatic retry on failures

## Authentication

The SDK supports two authentication modes:

1. **API Key Mode**: Direct authentication using your API key
2. **Delegation Token Mode**: Server-side token generation for enhanced security

Both modes support streaming responses.
