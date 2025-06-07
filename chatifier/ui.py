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
from prompt_toolkit.styles import Style

from .clients import BaseClient


console = Console()

# Define styles for different color modes
INPUT_STYLE = Style.from_dict({
    'user-input': '#808080',  # Light grey that works on both dark and light backgrounds
    'user-prompt': '#666666', # Slightly darker grey for the prompt
})

# Color for user text in panels (works on both light and dark)
USER_COLOR = "bright_black"  # This adapts to terminal theme


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
    content += "• [yellow]Ctrl+Enter[/yellow] - Multi-line input\n\n"
    content += "[dim]💡 The AI remembers conversation context. Use /clear to start fresh.[/dim]"
    
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
        
        # Get initial input with light grey color
        text = prompt("You: ", key_bindings=bindings, style=INPUT_STYLE)
        
        # Check if user wants multi-line input
        if text == 'multiline':
            console.print("[dim]Multi-line mode - press Ctrl+C when done:[/dim]")
            lines = []
            try:
                while True:
                    line = prompt("... ", style=INPUT_STYLE)
                    lines.append(line)
            except KeyboardInterrupt:
                text = '\n'.join(lines)
        
        return text.strip()
    
    except (KeyboardInterrupt, EOFError):
        return '/exit'


def display_user_message(message: str):
    """Display user message with subtle styling."""
    # Use bright_black which adapts to light/dark themes
    styled_message = f"[{USER_COLOR}]{message}[/{USER_COLOR}]"
    console.print(Panel(styled_message, title="👤 You", border_style="bright_black"))


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
            
            # Display user message
            display_user_message(user_input)
            
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
