#!/usr/bin/env python3
#
# module_crawler.py

import importlib
import inspect
from rich import inspect as isp 
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import sys
console = Console()
log = console.log
def generate_library_help(library_name):
    console = Console()

    try:
        # Dynamically import the library
        library = importlib.import_module(library_name)
    except ModuleNotFoundError as e:
        log(f"[red bold]Error:[/red bold] The library '{library_name}' could not be found.: {e}")
        return

    # Display Library Name
    title_panel = Panel(f"[bold cyan]Help Report for {library_name}[/bold cyan]", width=console.width)
    console.print(title_panel)

    # Display library docstring if available
    docstring = inspect.getdoc(library)
    if docstring:
        docstring_panel = Panel(docstring, title="Library Overview", width=console.width, style="green")
        console.print(docstring_panel)
    else:
        console.print("[yellow]No overview available for this library.[/yellow]")

    # Generate tables for classes and functions
    def add_class_table():
        table = Table(title="Classes and Methods", show_lines=True, width=console.width)
        table.add_column("Class", style="magenta", no_wrap=True)
        table.add_column("Method / Description", style="yellow")

        # Loop through classes in the library
        for class_name, cls in inspect.getmembers(library, inspect.isclass):
            if cls.__module__ == library.__name__:  # Filter only classes from the library
                class_doc = inspect.getdoc(cls) or "No description available"
                table.add_row(f"[bold]{class_name}[/bold]", class_doc)

                # Loop through methods within each class
                for method_name, method in inspect.getmembers(cls, inspect.isfunction):
                    # Get method docstring or parameter list if no docstring
                    method_doc = inspect.getdoc(method) or "Parameters: " + str(inspect.signature(method))
                    table.add_row(f"  • {method_name}()", method_doc)

        return table

    def add_function_table():
        table = Table(title="Functions", show_lines=True, width=console.width)
        table.add_column("Function", style="magenta", no_wrap=True)
        table.add_column("Description / Parameters", style="yellow")

        # Loop through functions in the library
        for func_name, func in inspect.getmembers(library, inspect.isfunction):
            # Get function docstring or parameter list if no docstring
            func_doc = inspect.getdoc(func) or "Parameters: " + str(inspect.signature(func))
            table.add_row(func_name, func_doc)

        return table

    # Recursively crawl modules, classes, and functions
    def crawl_module(module, prefix=""):
        # Create a table to display attributes and methods
        table = Table(show_lines=True, width=console.width)
        table.add_column("Attribute/Function", style="cyan", no_wrap=True)
        table.add_column("Description", style="yellow")

        # Recursively inspect each attribute in the module
        for name in dir(module):
            try:
                attr = getattr(module, name)
                if inspect.ismodule(attr) and attr.__name__.startswith(library_name):
                    # If it's a sub-module of the library, recursively crawl it
                    table.add_row(f"{prefix}{name}", "Sub-module")
                    table.add_row(f"{prefix}  └─ Crawling sub-module {name}...", "")
                    crawl_module(attr, prefix=prefix + "  ")
                elif inspect.isclass(attr):
                    # If it's a class, list its methods
                    class_doc = inspect.getdoc(attr) or "No description available"
                    table.add_row(f"{prefix}{name}", f"Class - {class_doc}")
                    for method_name, method in inspect.getmembers(attr, inspect.isfunction):
                        method_doc = inspect.getdoc(method) or "No description"
                        table.add_row(f"{prefix}  └─ {method_name}()", method_doc)
                elif inspect.isfunction(attr):
                    # If it's a function, add its description
                    func_doc = inspect.getdoc(attr) or "No description available"
                    table.add_row(f"{prefix}{name}()", func_doc)
                else:
                    # List other types of attributes (constants, variables)
                    table.add_row(f"{prefix}{name}", "Attribute or Variable")
            except Exception as e:
                # Handle any attributes that cannot be accessed
                table.add_row(f"{prefix}{name}", f"[red]Error: {e}[/red]")

        # Display the table if it has content
        if table.row_count > 0:
            console.print(table)

    # Display class and function tables
    class_table = add_class_table()
    function_table = add_function_table()

    if class_table.row_count > 0:
        console.print(class_table)
    else:
        console.print("[yellow]No classes available in this library.[/yellow]")

    if function_table.row_count > 0:
        console.print(function_table)
    else:
        console.print("[yellow]No functions available in this library.[/yellow]")

    # Recursively display attributes and submodules
    console.print("\n[bold cyan]Recursive Inspection of Module Attributes[/bold cyan]")
    crawl_module(library)

    # Section for example usage (if available in docstring)
    example_text = Text()
    if docstring and "Example" in docstring:
        example_text.append("\nExamples from library docstring:\n", style="bold green underline")
        example_text.append(docstring.split("Example")[1].strip(), style="yellow")
        console.print(example_text)
    else:
        console.print("[yellow]No examples found in library docstring.[/yellow]")

user_input = sys.argv[1]
generate_library_help(user_input)  # Replace "json" with any library name you want to inspect

isp(user_input, title=user_input, help=True, methods=True, all=True)
