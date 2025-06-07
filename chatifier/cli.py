#!/usr/bin/env python3
"""CLI interface for llm-chatifier."""

import logging
import sys
from typing import Optional

import click

from .detector import detect_api
from .clients import create_client
from .ui import run_chat
from .utils import prompt_for_token, parse_host_input


@click.command()
@click.argument('host', default='localhost')
@click.option('--port', '-p', type=int, help='Port number (overrides URL port)')
@click.option('--token', '-t', help='API token')
@click.option('--model', '-m', help='Model name to use')
@click.option('--override', '-o', help='Force specific API type (openai, ollama, anthropic, gemini, cohere, generic)')
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
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.debug(f"Starting chatifier with host={parsed_host}, port={final_port}, override={override}, model={model}")
    
    try:
        # 1. Detect or use override API type
        if override:
            api_info = {
                'type': override,
                'host': parsed_host,
                'port': final_port,
                'use_https': use_https,
                'base_url': None  # Will be constructed in create_client
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
        
        # 2. Create client
        client = create_client(api_info['type'], api_info.get('base_url'), token, model)
        
        # 3. Test connection and handle auth
        try:
            # Try to connect (this will be implemented in client)
            client.test_connection()
        except Exception as e:
            error_msg = str(e).lower()
            if 'auth' in error_msg or 'token' in error_msg or 'unauthorized' in error_msg:
                if not token:
                    click.echo("Authentication required.")
                    token = prompt_for_token()
                    client.token = token
                    try:
                        client.test_connection()
                    except Exception as e2:
                        click.echo(f"Authentication failed: {e2}")
                        sys.exit(1)
                else:
                    click.echo(f"Authentication failed: {e}")
                    sys.exit(1)
            else:
                click.echo(f"Connection failed: {e}")
                sys.exit(1)
        
        # 4. Handle model selection if no model specified
        if not client.model:
            try:
                models = client.get_models()
                if len(models) == 1:
                    client.model = models[0]
                    if verbose:
                        click.echo(f"Using model: {client.model}")
                else:
                    click.echo("Available models:")
                    for i, model_name in enumerate(models[:10], 1):  # Show max 10 models
                        click.echo(f"  {i}. {model_name}")
                    
                    if len(models) > 10:
                        click.echo(f"  ... and {len(models) - 10} more")
                    
                    while True:
                        try:
                            choice = click.prompt("Select model (number or name)", type=str)
                            
                            # Try as number first
                            try:
                                model_idx = int(choice) - 1
                                if 0 <= model_idx < len(models):
                                    client.model = models[model_idx]
                                    break
                            except ValueError:
                                pass
                            
                            # Try as model name
                            if choice in models:
                                client.model = choice
                                break
                            
                            click.echo("Invalid selection. Please try again.")
                        except KeyboardInterrupt:
                            click.echo("\nCancelled.")
                            sys.exit(0)
            except Exception as e:
                click.echo(f"Error: {e}")
                click.echo("Unable to retrieve models list. Please specify a model with --model flag.")
                sys.exit(1)
        
        # 5. Start chat UI
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
