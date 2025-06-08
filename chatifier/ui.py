"""Terminal UI for llm-chatifier."""

import sys
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from prompt_toolkit import prompt, PromptSession
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


def show_welcome(api_info: Dict[str, Any], client=None):
    """Display welcome message with API info."""
    api_type = api_info.get('type', 'unknown')
    base_url = api_info.get('base_url', 'unknown')
    
    title = f"🤖 llm-chatifier - Connected to {api_type.upper()}"
    content = f"Endpoint: [cyan]{base_url}[/cyan]\n"
    
    # Show model if available
    if client and hasattr(client, 'model') and client.model:
        content += f"Model: [green]{client.model}[/green]\n"
    
    content += "\n"
    content += "Commands:\n"
    content += "• [yellow]/exit[/yellow] or [yellow]Ctrl+C[/yellow] - Quit\n"
    content += "• [yellow]/clear[/yellow] - Clear conversation history (start fresh context)\n"
    content += "• [yellow]/help[/yellow] - Show this help\n"
    content += "• [yellow]Ctrl+J[/yellow] - Multi-line input\n\n"
    content += "[dim]💡 The AI remembers conversation context. Use /clear to start fresh.[/dim]"
    
    panel = Panel(content, title=title, border_style="green")
    console.print(panel)
    console.print()


def show_help():
    """Display help information."""
    help_text = """
[bold]Available Commands:[/bold]

• [yellow]/exit[/yellow] or [yellow]/quit[/yellow] - Exit the chat
• [yellow]/clear[/yellow] - Clear conversation history (start fresh context)
• [yellow]/help[/yellow] - Show this help message
• [yellow]Ctrl+C[/yellow] - Exit the chat
• [yellow]Ctrl+J[/yellow] - Enter multi-line input mode

[bold]Tips:[/bold]
• Just type your message and press Enter to chat
• Use Ctrl+J for multi-line messages
• In multi-line mode: Ctrl+D or Ctrl+X to submit
• The AI will remember the conversation context
"""
    console.print(Panel(help_text, title="Help", border_style="blue"))


def get_user_input(multiline_default: bool = False) -> str:
    """Get user input with support for multi-line and commands."""
    try:
        if multiline_default:
            # Start in multiline mode
            return get_multiline_input()
        
        # Set up key bindings for triggering multi-line input
        bindings = KeyBindings()
        
        @bindings.add('c-j')  # Ctrl+J (common alternative for Ctrl+Enter)
        def _(event):
            event.app.exit(result='multiline')
        
        # Get initial input with simple cursor and light grey color
        text = prompt("> ", key_bindings=bindings, style=INPUT_STYLE)
        
        # Check if user wants multi-line input
        if text == 'multiline':
            return get_multiline_input()
        
        return text.strip()
    
    except (KeyboardInterrupt, EOFError):
        return '/exit'


def get_multiline_input() -> str:
    """Get multi-line input with proper exit keys."""
    
    # Set up key bindings for multiline input
    bindings = KeyBindings()
    
    # Submit on Ctrl+D (EOF) - works on all platforms
    @bindings.add('c-d')
    def _(event):
        event.current_buffer.validate_and_handle()
    
    # Submit on Ctrl+X - alternative that works on Mac
    @bindings.add('c-x')
    def _(event):
        event.current_buffer.validate_and_handle()
    
    # Handle Ctrl+C (exit with empty)
    @bindings.add('c-c')
    def _(event):
        event.app.exit(exception=KeyboardInterrupt)
    
    console.print("[dim]Multi-line mode - press Ctrl+D or Ctrl+X when done:[/dim]")
    
    # Create a session for multiline input
    session = PromptSession(
        message="  ",
        multiline=True,
        key_bindings=bindings,
        style=INPUT_STYLE
    )
    
    try:
        text = session.prompt()
        return text.strip() if text else ""
    except (KeyboardInterrupt, EOFError):
        return ""


def display_response(response: str, render_markdown: bool = True):
    """Display AI response with optional markdown rendering."""
    try:
        if render_markdown and any(marker in response for marker in ['**', '*', '`', '#', '-', '1.', '```']):
            # Render as markdown
            md = Markdown(response)
            console.print(md)
        else:
            # Plain text response
            console.print(response)
    except Exception:
        # Fallback to plain text
        console.print(response)
    
    console.print()  # Add spacing after response


def display_error(error_msg: str):
    """Display error message."""
    console.print(Panel(f"[red]Error: {error_msg}[/red]", border_style="red"))


def display_thinking():
    """Show thinking indicator."""
    with console.status("[bold green]Thinking...", spinner="dots"):
        pass


def run_chat(client: BaseClient, api_info: Dict[str, Any], render_markdown: bool = True, multiline_default: bool = False):
    """Main chat loop.
    
    Args:
        client: API client instance
        api_info: Information about the detected API
        render_markdown: Whether to render markdown in responses
        multiline_default: Whether to use multiline input by default
    """
    # Show welcome message
    show_welcome(api_info, client)
    
    while True:
        try:
            # Get user input
            user_input = get_user_input(multiline_default)
            
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
            retry_count = 0
            max_retries = 1
            
            while retry_count <= max_retries:
                try:
                    with console.status("[bold green]Thinking...", spinner="dots"):
                        response = client.send_message(user_input)
                    
                    # Display response
                    display_response(response, render_markdown)
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    error_msg = str(e)
                    display_error(f"Error: {error_msg}")
                    
                    # If this is a retry, or auth error, don't auto-retry
                    if retry_count > 0 or 'auth' in error_msg.lower() or 'token' in error_msg.lower():
                        # If auth error, offer to retry with new token
                        if 'auth' in error_msg.lower() or 'token' in error_msg.lower():
                            if confirm("Would you like to enter a new token?"):
                                from .utils import prompt_for_token
                                client.token = prompt_for_token()
                                console.print("[green]Token updated. Please try your message again.[/green]")
                        else:
                            # For non-auth errors, restore the user's input
                            console.print(f"[yellow]Your message was: {user_input}[/yellow]")
                            console.print("[yellow]You can try rephrasing or use a different model.[/yellow]")
                        break
                    else:
                        # First attempt failed with non-auth error - retry once
                        retry_count += 1
                        if confirm(f"Request failed. Retry? (attempt {retry_count + 1}/{max_retries + 1})"):
                            console.print("[yellow]Retrying...[/yellow]")
                            continue
                        else:
                            # User chose not to retry - restore input
                            console.print(f"[yellow]Your message was: {user_input}[/yellow]")
                            break
        
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
