# llm-chatifier

Dead simple terminal chat client for LLM APIs.

## Features

- **Auto-detection**: Automatically detects API type by testing common endpoints
- **Multi-API support**: Works with OpenAI, Ollama, Anthropic, and generic chat APIs
- **Rich terminal UI**: Beautiful formatting with markdown support
- **Smart defaults**: Tries common ports and endpoints automatically
- **Minimal configuration**: Just provide an IP and optionally port/token

## Install

```bash
pipx install llm-chatifier
```

Or with pip:
```bash
pip install llm-chatifier
```

## Usage

```bash
# Auto-detect API on localhost
llm-chatifier

# Specify IP
llm-chatifier 192.168.1.100

# With port and token
llm-chatifier 192.168.1.100 --port 8080 --token sk-...

# Force specific API type
llm-chatifier --override openai

# Use specific model
llm-chatifier --model gpt-4

# Verbose output to see detection process
llm-chatifier -v
```

## Supported APIs

- **OpenAI compatible**: OpenAI, llama.cpp, vLLM, LocalAI, etc.
- **Ollama**: Local Ollama installations  
- **Anthropic**: Claude API
- **Google Gemini**: Gemini Pro and Vision models
- **Cohere**: Command and Command-R models
- **Generic**: Any chat API with common REST patterns

## Commands

Once connected, you can use these commands in the chat:

- `/exit` or Ctrl+C - Quit the application
- `/clear` - Clear conversation history  
- `/help` - Show help message
- Ctrl+Enter - Multi-line input mode

## How it Works

1. **Detection**: Tests common ports (8080, 8000, 3000, 5000, 11434, 80, 443) if no port specified
2. **Protocol**: Tries HTTPS first, falls back to HTTP
3. **Endpoints**: Tests API-specific endpoints to identify the service type
4. **Connection**: Creates appropriate client and tests authentication
5. **Chat**: Starts interactive terminal session

## API Detection

The tool automatically detects API types by testing these endpoints:

- **OpenAI**: `/v1/models`, `/v1/chat/completions`
- **Anthropic**: `/v1/messages`, `/v1/models`  
- **Ollama**: `/api/tags`, `/api/generate`
- **Gemini**: `/v1beta/models`, `/v1beta/models/gemini-pro:generateContent`
- **Cohere**: `/v1/chat`
- **Generic**: `/chat`, `/api/chat`, `/message`, `/api/message`

## Examples

```bash
# Local Ollama (usually on port 11434)
llm-chatifier localhost

# Remote OpenAI-compatible API
llm-chatifier my-server.com --port 8000 --token sk-xxx

# Force Ollama even if detection fails
llm-chatifier --override ollama

# Connect to Anthropic API
llm-chatifier --override anthropic --token your-claude-key

# Connect to Google Gemini
llm-chatifier --override gemini --token your-gemini-key

# Connect to Cohere
llm-chatifier --override cohere --token your-cohere-key
```

## Development

```bash
git clone https://github.com/fluffypony/llm-chatifier
cd llm-chatifier
pip install -e .
python -m chatifier --help
```

## License

BSD-3-Clause
