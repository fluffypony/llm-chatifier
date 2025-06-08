"""llm-chatifier: Universal terminal chat client for LLM APIs."""

try:
    from importlib.metadata import version
    __version__ = version("llm-chatifier")
except ImportError:
    # Fallback for older Python versions
    try:
        from importlib_metadata import version
        __version__ = version("llm-chatifier")
    except ImportError:
        __version__ = "unknown"
