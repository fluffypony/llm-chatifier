"""Terminal UI for llm-chatifier."""

import sys
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import confirm

from .clients import BaseClient


console = Console()


def show_welcome(api_info: Dict[str, Any]):
    """Display welcome message with API info."""
    api_type = api_info.get('type', 'unknown')
    base_url = api_info.get('base_url', 'unknown')
    
    title = f"🤖 llm-chatifier - Connected to {api_type.upper()}"
    content = f"Endpoint: [cyan]{base_url}[/cyan]\n\n"
    content += "Commands:\n"
    content += "• [yellow]/exit[/yellow] or [yellow]Ctrl+C[/yellow] - Quit\n"
    content += "• [yellow]/clear[/yellow] - Clear conversation history\n"
    content += "• [yellow]/help[/yellow] - Show this help\n"
    content += "• [yellow]Ctrl+Enter[/yellow] - Multi-line input"
    
    panel = Panel(content, title=title, border_style="green")
    console.print(panel)
    console.print()


def show_help():
    """Display help information."""
    help_text = """
[bold]Available Commands:[/bold]

• [yellow]/exit[/yellow] or [yellow]/quit[/yellow] - Exit the chat
• [yellow]/clear[/yellow] - Clear conversation history  
• [yellow]/help[/yellow] - Show this help message
• [yellow]Ctrl+C[/yellow] - Exit the chat
• [yellow]Ctrl+Enter[/yellow] - Enter multi-line input mode

[bold]Tips:[/bold]
• Just type your message and press Enter to chat
• Use Ctrl+Enter for multi-line messages
• The AI will remember the conversation context
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))


def get_user_input() -> str:
    """Get user input with support for multi-line and commands."""
    try:
        # Set up key bindings for multi-line input
        bindings = KeyBindings()
        
        @bindings.add('c-enter')  # Ctrl+Enter
        def _(event):
            event.app.exit(result='multiline')
        
        # Get initial input
        text = prompt("You: ", key_bindings=bindings)
        
        # Check if user wants multi-line input
        if text == 'multiline':
            console.print("[dim]Multi-line mode - press Ctrl+C when done:[/dim]")
            lines = []
            try:
                while True:
                    line = prompt("... ")
                    lines.append(line)
            except KeyboardInterrupt:
                text = '\n'.join(lines)
        
        return text.strip()
    
    except (KeyboardInterrupt, EOFError):
        return '/exit'


def display_response(response: str):
    """Display AI response with nice formatting."""
    try:
        # Try to render as markdown if it contains formatting
        if any(marker in response for marker in ['**', '*', '`', '#', '-', '1.']):
            md = Markdown(response)
            console.print(Panel(md, title="🤖 Assistant", border_style="blue"))
        else:
            # Plain text response
            console.print(Panel(response, title="🤖 Assistant", border_style="blue"))
    except Exception:
        # Fallback to plain text
        console.print(Panel(response, title="🤖 Assistant", border_style="blue"))


def display_error(error_msg: str):
    """Display error message."""
    console.print(Panel(f"[red]Error: {error_msg}[/red]", border_style="red"))


def display_thinking():
    """Show thinking indicator."""
    with console.status("[bold green]Thinking...", spinner="dots"):
        pass


def run_chat(client: BaseClient, api_info: Dict[str, Any]):
    """Main chat loop.
    
    Args:
        client: API client instance
        api_info: Information about the detected API
    """
    # Show welcome message
    show_welcome(api_info)
    
    while True:
        try:
            # Get user input
            user_input = get_user_input()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['/exit', '/quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            elif user_input.lower() == '/clear':
                client.clear_history()
                console.print("[green]Conversation history cleared.[/green]")
                continue
            
            elif user_input.lower() == '/help':
                show_help()
                continue
            
            # Send message to API
            try:
                with console.status("[bold green]Thinking...", spinner="dots"):
                    response = client.send_message(user_input)
                
                # Display response
                display_response(response)
                console.print()  # Add spacing
                
            except Exception as e:
                display_error(str(e))
                
                # If auth error, offer to retry with new token
                if 'auth' in str(e).lower() or 'token' in str(e).lower():
                    if confirm("Would you like to enter a new token?"):
                        from .utils import prompt_for_token
                        client.token = prompt_for_token()
                        console.print("[green]Token updated. Please try again.[/green]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        
        except Exception as e:
            display_error(f"Unexpected error: {e}")
            break


def show_connection_status(host: str, port: int, success: bool):
    """Show connection attempt status."""
    status = "[green]✓[/green]" if success else "[red]✗[/red]"
    console.print(f"{status} Trying {host}:{port}")


def show_detection_progress(api_type: str, endpoint: str, success: bool):
    """Show API detection progress."""
    status = "[green]✓[/green]" if success else "[dim]✗[/dim]"
    console.print(f"{status} Testing {api_type}: {endpoint}")
