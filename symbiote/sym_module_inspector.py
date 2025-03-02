#!/usr/bin/env python3
#
# sym_module_inspector.py

from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
console = Console()
log = console.log
print = console.print
import importlib
import inspect
import html
import json
import yaml
import sys

class ModuleInspector:
    def __init__(self, library_name):
        parts = library_name.split(".")
        self.library_name = library_name 
        self.library_last = parts.pop()
        self.library = None
        self.output_data = {}
        self.visited = set()  # Track visited components by their id to avoid excessive recursion

    def load_library(self):
        """Loads the library, handling cases where the module is not found."""
        try:
            self.library = importlib.import_module(self.library_name)
        except ModuleNotFoundError:
            console.print(f"[red bold]Error:[/red bold] The library '{self.library_name}' could not be found.")
            return False
        return True

    def generate_report(self, output="dir", sub=False):
        """Generates a report of the library, including classes, functions, and attributes, in the specified format."""
        # Load the library
        if not self.load_library():
            return

        self.output_data = {
            "library": self.library_name,
            "attributes": self.crawl_module(self.library)
        }

        # Output the data in the specified format
        if output == "json":
            return json.dumps(self.output_data, indent=4)
        elif output == "yaml":
            return yaml.dump(self.output_data, default_flow_style=False)
        elif output == "markdown":
            return self.to_markdown()
        elif output == "html":
            return self.to_html()
        elif output == "dir":
            return self.output_data
        elif output == "render":
            self.render_report()
            #return self.output_data

    def render_report(self):
        """Renders the report to the console using rich components."""
        #if not self.output_data:
        #    self.generate_report()
        with console.pager(styles=True):
            # Display Library Name
            title_panel = Panel(f"[bold cyan]Help Report for {self.library_name}[/bold cyan]", width=console.width)
            console.print(title_panel)

            # Render the recursive attributes tree
            tree_table = self._render_recursive_attributes(self.output_data["attributes"])
            console.print(Panel(tree_table))

    def crawl_module(self, module, prefix=""):
        """Recursively crawls module attributes and gathers data for JSON output, with protection against excessive recursion."""
        attributes = []

        for name in dir(module):
            try:
                attr = getattr(module, name)
                attr_id = id(attr)
                if attr_id in self.visited:
                    continue  # Skip already visited attributes to prevent excessive recursion

                self.visited.add(attr_id)

                if inspect.ismodule(attr) and attr.__name__.startswith(self.library_name):
                    attributes.append({
                        "module": name,
                        "type": "Sub-module",
                        "attributes": self.crawl_module(attr, prefix=prefix + "  ")
                    })
                elif inspect.isclass(attr):
                    class_doc = inspect.getdoc(attr) or "No description available"
                    methods = [
                        {"method": method_name, "docstring": inspect.getdoc(method) or "No description"}
                        for method_name, method in inspect.getmembers(attr, inspect.isfunction)
                        if id(method) not in self.visited
                    ]
                    for method in methods:
                        self.visited.add(id(method))
                    attributes.append({"class": name, "docstring": class_doc, "methods": methods})
                elif inspect.isfunction(attr):
                    func_doc = inspect.getdoc(attr) or "No description available"
                    attributes.append({"function": name, "docstring": func_doc})
                else:
                    attributes.append({"attribute": name, "type": "Attribute or Variable"})
            except Exception as e:
                attributes.append({"attribute": name, "error": str(e)})

        return attributes

    def _create_class_table(self):
        """Creates a rich Table for classes and their methods."""
        table = Table(title="Classes and Methods", show_lines=True, width=console.width)
        table.add_column("Class", style="magenta", no_wrap=True)
        table.add_column("Method / Description", style="yellow")

        for class_info in self.output_data["classes"]:
            table.add_row(f"[bold]{class_info['class']}[/bold]", class_info["docstring"])

            for method in class_info[":methods"]:
                method_name = method["method"]
                method_doc = method["docstring"]
                table.add_row(f"  • {method_name}()", method_doc)

        return table

    def _create_function_table(self):
        """Creates a rich Table for functions in the library."""
        table = Table(title="Functions", show_lines=True, width=console.width)
        table.add_column("Function", style="magenta", no_wrap=True)
        table.add_column("Description / Parameters", style="yellow")

        for func_info in self.output_data["functions"]:
            func_name = func_info["function"]
            func_doc = func_info["docstring"]
            table.add_row(func_name, func_doc)

        return table

    def _render_recursive_attributes(self, attributes, path_prefix=""):
        """Recursively renders attributes with callable paths, nested tree structures, tables, and color-matched panels."""
        root_tree = Tree(f"[bold cyan]Attributes for {self.library_name}[/bold cyan]")

        def render_attr_tree(attr_list, parent_tree, current_path):
            for attr in attr_list:
                if "module" in attr:
                    # Module block with optional docstring panel
                    module_path = f"{current_path}.{attr['module']}"
                    module_tree = parent_tree.add(f"[bold blue]Module[/bold blue]: {attr['module']} [italic][{module_path}][/italic]")
                    if attr.get("docstring"):
                        module_tree.add(Panel(attr["docstring"], title="Module Description", border_style="blue"))
                    render_attr_tree(attr["attributes"], module_tree, module_path)
                elif "class" in attr:
                    # Class block with callable path and description
                    class_path = f"{current_path}.{attr['class']}"
                    class_tree = parent_tree.add(f"[bold magenta]Class[/bold magenta]: {attr['class']} [italic][{class_path}][/italic]")
                    
                    # Class description panel if available
                    if attr.get("docstring"):
                        class_tree.add(Panel(attr["docstring"], title="Class Description", border_style="magenta"))
                    
                    # Methods table with visible row lines
                    if attr["methods"]:
                        methods_table = Table(show_header=True, header_style="bold yellow", show_lines=True)
                        methods_table.add_column("Method", style="green", no_wrap=True)
                        methods_table.add_column("Description", style="dim")

                        for method in attr["methods"]:
                            method_path = f"{class_path}.{method['method']}"
                            methods_table.add_row(f"{method['method']} [italic][{method_path}][/italic]", method["docstring"])

                        # Set the Methods panel to a green border
                        class_tree.add(Panel(methods_table, title="Methods", border_style="green"))
                elif "function" in attr:
                    # Function block with callable path and docstring panel if available
                    function_path = f"{current_path}.{attr['function']}"
                    function_tree = parent_tree.add(f"[cyan]Function[/cyan]: {attr['function']} [italic][{function_path}][/italic]")
                    if attr.get("docstring"):
                        function_tree.add(Panel(attr["docstring"], title="Function Description", border_style="cyan"))
                elif "attribute" in attr:
                    # Attribute block with type
                    parent_tree.add(f"[yellow]Attribute[/yellow]: {attr['attribute']} - {attr['type']}")

        # Populate the tree structure with attributes and their callable paths
        render_attr_tree(attributes, root_tree, path_prefix or self.library_name)
        return root_tree
        # Render the full tree in the console
        #console.print(root_tree)

    def to_html(self):
        """Converts output data to a styled HTML format matching the 'render' theme."""
        # HTML template with color-coded styles for each section
        html_output = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    line-height: 1.5;
                }}
                h1, h2, h3, h4 {{
                    color: #569cd6;
                }}
                .module h2 {{
                    color: #4ec9b0;
                }}
                .class h3 {{
                    color: #dcdcaa;
                }}
                .function h3 {{
                    color: #569cd6;
                }}
                .attribute h3 {{
                    color: #9cdcfe;
                }}
                .method h4 {{
                    color: #c586c0;
                }}
                .panel {{
                    padding: 0px 10px 10px 20px;
                    border: 1px solid #444;
                    border-radius: 5px;
                    margin: 10px 0;
                    background-color: #2d2d2d;
                }}
                .panel.module {{
                    border-color: #4ec9b0;
                }}
                .panel.class {{
                    border-color: #dcdcaa;
                }}
                .panel.method {{
                    border-color: #c586c0;
                }}
                .panel.function {{
                    border-color: #569cd6;
                }}
                .attribute {{
                    color: #9cdcfe;
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                .table th, .table td {{
                    padding: 8px;
                    border: 1px solid #444;
                    text-align: left;
                }}
                .table th {{
                    background-color: #3b3b3b;
                    color: #9cdcfe;
                }}
                .indent-1 {{ margin-left: 20px; }}
                .indent-2 {{ margin-left: 40px; }}
                .indent-3 {{ margin-left: 60px; }}
                .indent-4 {{ margin-left: 80px; }}
                .library-name {{ color: #FFFFFF; }}
            </style>
        </head>
        <body>
            <h1><span class="library-name">{html.escape(self.library_name)}</span></h1>
            {self._render_recursive_attributes_html(self.output_data["attributes"], self.library_name, ["library"])}
        </body>
        </html>
        """
        return html_output

    def _render_recursive_attributes_html(self, attributes, path_prefix, path_types, indent_level=1):
        """Recursively generates HTML for attributes in the output data with nested panels and color-coded paths."""
        html_content = ""
        
        for attr in attributes:
            indent_class = f"indent-{indent_level}"
            
            if "module" in attr:
                module_name = self._format_colored_path(path_prefix, attr['module'], path_types + ["module"])
                html_content += f"""
                <div class="panel module {indent_class}">
                    <h2>Module: {module_name}</h2>
                    <div class="panel">{html.escape(attr.get("docstring", "No description available"))}</div>
                    {self._render_recursive_attributes_html(attr["attributes"], f"{path_prefix}.{attr['module']}", path_types + ["module"], indent_level + 1)}
                </div>
                """
            elif "class" in attr:
                class_name = self._format_colored_path(path_prefix, attr['class'], path_types + ["class"])
                html_content += f"""
                <div class="panel class {indent_class}">
                    <h3>Class: {class_name}</h3>
                    <div class="panel">{html.escape(attr.get("docstring", "No description available"))}</div>
                    {self._render_methods_table_html(attr["methods"], f"{path_prefix}.{attr['class']}", path_types + ["class"], indent_level + 2)}
                </div>
                """
            elif "function" in attr:
                function_name = self._format_colored_path(path_prefix, attr['function'], path_types + ["function"])
                html_content += f"""
                <div class="panel function {indent_class}">
                    <h3>Function: {function_name}</h3>
                    <div class="panel">{html.escape(attr.get("docstring", "No description available"))}</div>
                </div>
                """
            elif "method" in attr:
                method_name = self._format_colored_path(path_prefix, attr['method'], path_types + ["method"])
                html_content += f"""
                <div class="panel method {indent_class}">
                    <h4>Method: {method_name}</h4>
                    <div class="panel">{html.escape(attr.get("docstring", "No description available"))}</div>
                </div>
                """
            elif "attribute" in attr:
                # No path prefix for attributes
                html_content += f"""
                <div class="panel attribute {indent_class}">
                    <p><strong>Attribute:</strong> {html.escape(attr['attribute'])} - {html.escape(attr['type'])}</p>
                </div>
                """

        return html_content

    def _render_methods_table_html(self, methods, class_path, path_types, indent_level):
        """Generates an HTML table for class methods with a bordered panel and prefixed path for each method."""
        if not methods:
            return ""

        indent_class = f"indent-{indent_level}"

        table_html = f"""
        <div class="panel method {indent_class}">
            <h4 >Methods:</h4>
            <table class="table">
                <tr><th>Method</th><th>Description</th></tr>
        """
        
        for method in methods:
            method_name = self._format_colored_path(class_path, method['method'], path_types + ["method"])
            table_html += f"<tr><td>{method_name}</td><td>{html.escape(method['docstring'])}</td></tr>"

        table_html += "</table></div>"
        return table_html

    def _format_colored_path(self, path_prefix, name, path_types):
        """Generates a color-coded HTML path with the fixed library prefix in red and subsequent components styled by type."""
        path_parts = path_prefix.split(".")
        formatted_path = f"<span class='library-name'>{html.escape(self.library_last)}.</span>"

        # Define color map for each part type
        color_map = {
            "library": "#FFFFFF",
            "module": "#4ec9b0",
            "class": "#dcdcaa",
            "function": "#569cd6",
            "method": "#c586c0"
        }

        # Apply colors to each part of the path after the fixed library prefix
        for i, part in enumerate(path_parts[len(self.library_name.split(".")):]):
            part_type = path_types[i + len(self.library_last.split("."))] if i < len(path_types) else "module"
            formatted_path += f"<span style='color: {color_map.get(part_type, '#d4d4d4')};'>{html.escape(part)}.</span>"

        # Add the current name in its appropriate color
        current_type = path_types[-1]
        formatted_path += f"<span style='color: {color_map.get(current_type, '#d4d4d4')};'>{html.escape(name)}</span>"

        return formatted_path

    def __del__(self):
        """Cleans up resources when the object is deleted."""
        # Clear the output data to free up memory
        self.output_data.clear()

        # Clear the visited set
        self.visited.clear()

# Example Usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python module_inspector.py <module_name> [output_format]")
    else:
        module_name = sys.argv[1]
        output_format = sys.argv[2] if len(sys.argv) > 2 else "dir"
        inspector = ModuleInspector(module_name)
        report = inspector.generate_report(output=output_format)
