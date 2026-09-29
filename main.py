"""Entry point:  python main.py"""

from cli.app import CLIApp
from cli.display import console


def main() -> None:
    try:
        CLIApp().run()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold cyan]Goodbye![/bold cyan]")


if __name__ == "__main__":
    main()
