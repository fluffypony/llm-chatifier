#!/usr/bin/env python3
"""CLI interface for llm-chatifier."""

import logging
import sys
from typing import Optional

import click

from .detector import detect_api
from .clients import create_client
from .ui import run_chat
from .utils import prompt_for_token, parse_host_input, build_base_url, find_api_key_in_env


def select_model_from_list(models: list[str]) -> Optional[str]:
    """Select a model from a list displayed in columns."""
    if not models:
        return None
    
    if len(models) == 1:
        return models[0]
    
    # Reverse list to show newest models first
    models = list(reversed(models))
    
    # Show all models
    click.echo(f"Available models ({len(models)} total, newest first):")
    click.echo()
    
    # Display in 5 columns
    cols = 5
    rows = (len(models) + cols - 1) // cols
    
    for row in range(rows):
        line_parts = []
        for col in range(cols):
            idx = row + col * rows
            if idx < len(models):
                num = idx + 1
                model_name = models[idx]
                # Truncate long model names to fit in columns
                if len(model_name) > 15:
                    model_name = model_name[:12] + "..."
                line_parts.append(f"{num:2d}. {model_name:<15}")
            else:
                line_parts.append(" " * 19)
        click.echo("  ".join(line_parts))
    
    click.echo()
    click.echo("Select a model by number, or 'q' to quit:")
    
    while True:
        try:
            choice = click.prompt("Model", type=str, show_default=False)
            
            if choice.lower() == 'q':
                return None
            
            try:
                model_idx = int(choice) - 1
                if 0 <= model_idx < len(models):
                    return models[model_idx]
                else:
                    click.echo(f"Please enter a number between 1 and {len(models)}")
            except ValueError:
                # Try to find model by name
                if choice in models:
                    return choice
                click.echo("Invalid selection. Please enter a number or 'q' to quit.")
                
        except KeyboardInterrupt:
            return None


@click.command()
@click.argument('host', default='localhost')
@click.option('--port', '-p', type=int, help='Port number (overrides URL port)')
@click.option('--token', '-t', help='API token')
@click.option('--model', '-m', help='Model name to use')
@click.option('--override', '-o', help='Force specific API type (openai, openrouter, ollama, anthropic, gemini, cohere, generic)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--no-markdown', is_flag=True, help='Disable markdown rendering')
@click.option('--multiline', is_flag=True, help='Use multiline input by default')
def main(host: str, port: Optional[int], token: Optional[str], model: Optional[str], override: Optional[str], verbose: bool, no_markdown: bool, multiline: bool):
    """Simple chat client for LLM APIs.
    
    HOST: IP address, hostname, or full URL (default: localhost)
    """
    # Parse host input (could be IP, hostname, or full URL)
    parsed_host, parsed_port, use_https = parse_host_input(host)
    
    # Port override: CLI --port takes precedence over URL port
    final_port = port if port is not None else parsed_port
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    try:
        # 1. Detect or use override API type
        if override:
            # Construct base URL from user input
            if final_port:
                base_url = build_base_url(parsed_host, final_port, use_https)
            else:
                base_url = f"{'https' if use_https else 'http'}://{parsed_host}"
            
            api_info = {
                'type': override,
                'host': parsed_host,
                'port': final_port,
                'use_https': use_https,
                'base_url': base_url
            }
            if verbose:
                click.echo(f"Using override API type: {override}")
        else:
            if verbose:
                click.echo(f"Auto-detecting API on {parsed_host}...")
            api_info = detect_api(parsed_host, final_port, use_https, token)
            if not api_info:
                click.echo(f"Error: No compatible API found on {parsed_host}")
                if final_port:
                    click.echo(f"Tried port {final_port}")
                else:
                    click.echo("Tried common ports: 8080, 8000, 3000, 5000, 11434, 80, 443")
                click.echo("Use --override to force a specific API type")
                sys.exit(1)
            
            if verbose:
                click.echo(f"Detected {api_info['type']} API at {api_info['base_url']}")
        
        # 2. Test API without auth first to see if auth is needed
        client = create_client(api_info['type'], api_info.get('base_url'), None, model, verbose)
        
        if verbose:
            click.echo("Testing API without authentication...")
        
        try:
            response = client.test_with_simple_prompt()
            if verbose:
                click.echo(f"API test successful without auth: {response[:50]}...")
            # No auth needed - we're done!
            
        except Exception as e:
            error_msg = str(e).lower()
            # Broader definition of "needs auth" - includes JSON parsing errors, empty responses, etc.
            is_auth_error = (
                'auth' in error_msg or 'token' in error_msg or 'unauthorized' in error_msg or 'api key' in error_msg or
                'expecting value' in error_msg or  # JSON parsing errors often mean auth issues
                'empty response' in error_msg or
                'no content' in error_msg
            )
            
            if is_auth_error:
                if verbose:
                    click.echo("Authentication required.")
                
                # 3. Get auth details - try environment first, then prompt
                working_token = None
                if not token:
                    from .utils import find_all_api_keys_in_env
                    env_tokens = find_all_api_keys_in_env(api_info['type'], api_info.get('base_url', ''))
                    
                    if env_tokens:
                        if verbose:
                            click.echo(f"Found {len(env_tokens)} potential API keys in environment")
                        
                        # 4. Try test prompt with each auth token
                        for i, env_token in enumerate(env_tokens):
                            if verbose:
                                click.echo(f"Trying API key {i+1}/{len(env_tokens)}...")
                            
                            client.token = env_token
                            try:
                                response = client.test_with_simple_prompt()
                                if verbose:
                                    click.echo(f"Authentication successful: {response[:50]}...")
                                working_token = env_token
                                break
                            except Exception as auth_e:
                                error_msg = str(auth_e).lower()
                                
                                # Check if this is a model requirement error (auth worked, but model needed)
                                if any(phrase in error_msg for phrase in ['no models provided', 'model required', 'must specify model']):
                                    if verbose:
                                        click.echo(f"Key {i+1} authenticated but requires model specification")
                                    working_token = env_token
                                    # Don't break - we'll handle model selection later
                                    break
                                
                                # Check if this is an auth error (try next key)
                                elif any(phrase in error_msg for phrase in ['auth', 'token', 'unauthorized', 'api key', 'invalid jwt', 'credentials']):
                                    if verbose:
                                        click.echo(f"Key {i+1} authentication failed: {str(auth_e)[:100]}...")
                                    continue
                                
                                # Other errors (network, etc.) - stop trying
                                else:
                                    if verbose:
                                        click.echo(f"API error with key {i+1}: {str(auth_e)[:100]}...")
                                    click.echo(f"API error: {auth_e}")
                                    sys.exit(1)
                
                # If no env tokens worked, prompt for manual input
                if not working_token:
                    if not token:
                        click.echo("Please provide an API key:")
                        token = prompt_for_token()
                    
                    client.token = token
                    try:
                        response = client.test_with_simple_prompt()
                        if verbose:
                            click.echo(f"Authentication successful: {response[:50]}...")
                        working_token = token
                    except Exception as manual_auth_e:
                        # 6. If test prompt still fails, try getting models list
                        if verbose:
                            click.echo("Test prompt failed, trying to get models list...")
                        
                        try:
                            models = client.get_models()
                            if models:
                                if verbose:
                                    click.echo(f"Found {len(models)} models: {', '.join(models[:3])}...")
                                # TODO: Let user select model and retry
                                click.echo("API accessible but default model may not work. Try specifying a model with -m")
                            else:
                                click.echo(f"Authentication failed: {manual_auth_e}")
                                sys.exit(1)
                        except Exception as models_e:
                            click.echo(f"Authentication failed: {manual_auth_e}")
                            sys.exit(1)
                            
            else:
                # Non-auth error - could be model issue
                if 'model' in error_msg and model:
                    if verbose:
                        click.echo(f"Model '{model}' failed, trying to get available models...")
                    
                    # 6. Try getting models list to help user
                    try:
                        models = client.get_models()
                        if models:
                            click.echo(f"Model '{model}' not available. Available models: {', '.join(models[:5])}")
                            sys.exit(1)
                    except:
                        pass
                
                click.echo(f"API error: {e}")
                sys.exit(1)
        
        # 5. Handle model selection if no model specified
        if not client.model:
            try:
                # Use cached models from the API test if available
                if hasattr(client, '_cached_models') and client._cached_models:
                    models = client._cached_models
                    if verbose:
                        click.echo("Using cached models from API test.")
                else:
                    models = client.get_models()
                
                if len(models) == 0:
                    click.echo("No models available from API. Please specify a model with --model flag.")
                    click.echo("Example: python -m chatifier --model claude-3-sonnet-20240229")
                    sys.exit(1)
                elif len(models) == 1:
                    client.model = models[0]
                    if verbose:
                        click.echo(f"Using model: {client.model}")
                else:
                    selected_model = select_model_from_list(models)
                    if selected_model:
                        client.model = selected_model
                        if verbose:
                            click.echo(f"Selected model: {client.model}")
                    else:
                        click.echo("No model selected.")
                        sys.exit(0)
            except Exception as e:
                click.echo(f"Error: {e}")
                click.echo("Unable to retrieve models list. Please specify a model with --model flag.")
                sys.exit(1)
        
        # 6. Start chat UI
        render_markdown = not no_markdown
        run_chat(client, api_info, render_markdown, multiline)
        
    except KeyboardInterrupt:
        click.echo("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        if verbose:
            raise
        click.echo(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
