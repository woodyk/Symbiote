#!/usr/bin/env python3
#
# richLogging.py

import logging
from rich.logging import RichHandler
from rich.console import Console

# Initialize Rich console
console = Console()

# Utility function to add a rule and pause
def rule_and_pause(title: str):
    console.print("\n\n")
    console.rule(title)

# 1. **Basic Logging**
rule_and_pause("Example 1: Basic Logging")
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, handlers=[RichHandler()]
)
log = logging.getLogger("rich")
log.info("This is an [bold green]info[/bold green] message.")
log.warning("This is a [yellow]warning[/yellow].")
log.error("[bold red]Something went wrong![/bold red]")

# 2. **Enable Console Markup in Logs**
rule_and_pause("Example 2: Enable Console Markup in Logs")
rich_handler = RichHandler(markup=True)
logging.basicConfig(handlers=[rich_handler])
log.info("[bold magenta]This message uses Rich markup[/]")

# 3. **Highlight Keywords**
rule_and_pause("Example 3: Highlight Keywords")
rich_handler = RichHandler(keywords=["IMPORTANT", "CRITICAL", "DEBUG"])
logging.basicConfig(handlers=[rich_handler])
log.info("This is an IMPORTANT log message.")
log.debug("CRITICAL debug log.")

# 4. **Customized Logging Format**
rule_and_pause("Example 4: Customized Logging Format")
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level="DEBUG", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log.debug("This is a debug message with a detailed format.")

# 5. **Using Rich Tracebacks**
rule_and_pause("Example 5: Using Rich Tracebacks")
try:
    1 / 0
except ZeroDivisionError:
    log.exception("Caught an exception!", extra={"rich_tracebacks": True})

# 6. **Suppressing External Library Frames**
rule_and_pause("Example 6: Suppressing External Library Frames")
import click

logging.basicConfig(
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            tracebacks_suppress=[click],  # Suppress Click frames in tracebacks
        )
    ]
)
try:
    raise ValueError("Test error in Click")
except ValueError:
    log.exception("An error occurred!")

# 7. **Dynamic Highlighting for Specific Messages**
rule_and_pause("Example 7: Dynamic Highlighting for Specific Messages")
log.error("123 will not be highlighted", extra={"highlighter": None})
log.error("[red]Error code: 123[/red]", extra={"markup": True})

# 8. **Rich Log with Multiple Handlers**
rule_and_pause("Example 8: Rich Log with Multiple Handlers")
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)
logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    handlers=[RichHandler(), file_handler],
)
log.info("This log is both printed in the console and written to a file.")

# 9. **Disable Timestamp or Level**
rule_and_pause("Example 9: Disable Timestamp or Level")
logging.basicConfig(
    level="DEBUG",
    handlers=[RichHandler(show_time=False, show_level=False)]
)
log.info("Log without time or level column.")

# 10. **Alternate Log Levels and Linking to File Paths**
rule_and_pause("Example 10: Alternate Log Levels and Linking to File Paths")
rich_handler = RichHandler(enable_link_path=True)
logging.basicConfig(handlers=[rich_handler])
log.debug("Debugging: Check this file path.")

# 11. **Handling Log Exceptions Gracefully**
rule_and_pause("Example 11: Handling Log Exceptions Gracefully")
try:
    raise RuntimeError("Critical failure in system.")
except RuntimeError as e:
    log.exception("Handled runtime error gracefully.")

# 12. **Combining Rich Log with Progress Display**
rule_and_pause("Example 12: Combining Rich Log with Progress Display")
from rich.progress import track

for _ in track(range(5), description="Processing..."):
    log.info("Progress step completed.")

