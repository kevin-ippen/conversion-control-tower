"""
CLI Entry Point for Conversion Control Tower

Usage:
    python -m cli.convert analyze --source <path>
    python -m cli.convert convert-ssis --source <path> --output <dir>
    python -m cli.convert config show
    python -m cli.convert examples list
"""

import click
import json
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

console = Console()


# Global config path option
def get_config_path(ctx) -> Path:
    """Get config path from context or default."""
    return ctx.obj.get("config_path") if ctx.obj else Path(__file__).parent.parent / "config"


@click.group()
@click.option(
    "--config-dir", "-c",
    type=click.Path(exists=True),
    default=None,
    help="Path to configuration directory"
)
@click.option(
    "--catalog",
    default=None,
    help="Override default Unity Catalog name"
)
@click.option(
    "--schema",
    default=None,
    help="Override default schema name"
)
@click.version_option(version="0.1.0")
@click.pass_context
def cli(ctx, config_dir: str, catalog: str, schema: str):
    """Conversion Control Tower.

    Convert SQL Server, SSIS, and Informatica PowerCenter workloads to
    Databricks PySpark/DLT/dbt with AI-powered semantic understanding.

    Use --config-dir to specify custom configuration.
    Use --catalog and --schema to override Unity Catalog settings.
    """
    ctx.ensure_object(dict)

    if config_dir:
        ctx.obj["config_path"] = Path(config_dir)
    else:
        ctx.obj["config_path"] = Path(__file__).parent.parent / "config"

    ctx.obj["catalog_override"] = catalog
    ctx.obj["schema_override"] = schema


# =============================================================================
# CONFIGURATION COMMANDS
# =============================================================================

@cli.group()
def config():
    """Manage conversion configuration."""
    pass


@config.command("show")
@click.option("--section", "-s", default=None, help="Show specific section only")
@click.pass_context
def config_show(ctx, section: str):
    """Display current configuration settings."""
    from src.config_loader import ConfigLoader

    loader = ConfigLoader(get_config_path(ctx))
    config = loader.load_config()

    if section:
        _show_config_section(config, section)
    else:
        _show_full_config(config)


def _show_full_config(config):
    """Display full configuration."""
    console.print(Panel(
        f"[bold]{config.config_name}[/bold] v{config.version}",
        title="Configuration"
    ))

    # Target settings
    table = Table(title="Target Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Platform", config.target_platform)
    table.add_row("Runtime Version", config.runtime_version)
    table.add_row("Language", config.language)
    table.add_row("Unity Catalog", f"{config.default_catalog}.{config.default_schema}")
    table.add_row("Delta Lake", "Enabled" if config.delta_enabled else "Disabled")
    table.add_row("Secrets Scope", config.secrets_scope)

    console.print(table)
    console.print()

    # Examples loaded
    console.print(f"[bold]Examples loaded:[/bold] {len(config.examples)}")
    for ex in config.examples[:5]:
        console.print(f"  - {ex.name}: {ex.description[:50]}...")
    if len(config.examples) > 5:
        console.print(f"  ... and {len(config.examples) - 5} more")


def _show_config_section(config, section: str):
    """Display specific configuration section."""
    section_map = {
        "target": ["target_platform", "runtime_version", "language"],
        "unity_catalog": ["unity_catalog_enabled", "default_catalog", "default_schema", "use_three_part_names"],
        "delta": ["delta_enabled", "clustering_strategy"],
        "secrets": ["secrets_provider", "secrets_scope", "enforce_secrets"],
        "output": ["include_source_comments", "add_todo_markers", "generate_docs"],
    }

    if section not in section_map:
        console.print(f"[red]Unknown section: {section}[/red]")
        console.print(f"Available sections: {', '.join(section_map.keys())}")
        return

    table = Table(title=f"Configuration: {section}")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    for attr in section_map[section]:
        value = getattr(config, attr, "N/A")
        table.add_row(attr, str(value))

    console.print(table)


@config.command("validate")
@click.pass_context
def config_validate(ctx):
    """Validate configuration file."""
    from src.config_loader import ConfigLoader

    try:
        loader = ConfigLoader(get_config_path(ctx))
        config = loader.load_config()
        console.print("[green]✓[/green] Configuration is valid")
        console.print(f"  - {len(config.examples)} examples loaded")
        console.print(f"  - {len(config.sql_transforms)} SQL transforms defined")
    except Exception as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")


@config.command("init")
@click.argument("output_dir", type=click.Path(), default=".")
def config_init(output_dir: str):
    """Initialize a new configuration in the specified directory."""
    import shutil

    output_path = Path(output_dir) / "config"
    source_path = Path(__file__).parent.parent / "config"

    if output_path.exists():
        console.print(f"[yellow]Config directory already exists at {output_path}[/yellow]")
        return

    shutil.copytree(source_path, output_path)
    console.print(f"[green]✓[/green] Created configuration at {output_path}")
    console.print("  Edit conversion_config.yaml to customize settings")


# =============================================================================
# EXAMPLES COMMANDS
# =============================================================================

@cli.group()
def examples():
    """Manage conversion examples."""
    pass


@examples.command("list")
@click.option("--tags", "-t", multiple=True, help="Filter by tags")
@click.pass_context
def examples_list(ctx, tags: tuple):
    """List all available conversion examples."""
    from src.config_loader import ConfigLoader

    loader = ConfigLoader(get_config_path(ctx))
    examples_list = loader.load_examples(list(tags) if tags else None)

    table = Table(title="Conversion Examples")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Tags", style="green")
    table.add_column("Description", max_width=40)

    for ex in examples_list:
        table.add_row(
            ex.name,
            ex.component_type or ex.input_type,
            ", ".join(ex.tags[:3]),
            ex.description[:40] + "..." if len(ex.description) > 40 else ex.description
        )

    console.print(table)


@examples.command("show")
@click.argument("name")
@click.pass_context
def examples_show(ctx, name: str):
    """Show details of a specific example."""
    from src.config_loader import ConfigLoader

    loader = ConfigLoader(get_config_path(ctx))
    examples_list = loader.load_examples()

    example = next((e for e in examples_list if e.name == name), None)

    if not example:
        console.print(f"[red]Example not found: {name}[/red]")
        return

    console.print(Panel(
        f"[bold]{example.name}[/bold]\n{example.description}",
        title="Example"
    ))

    console.print(f"\n[bold]Tags:[/bold] {', '.join(example.tags)}")
    console.print(f"[bold]Component Type:[/bold] {example.component_type or 'N/A'}")

    console.print("\n[bold cyan]Input:[/bold cyan]")
    console.print(Syntax(example.input_content, "xml" if "xml" in example.input_type else "sql"))

    console.print("\n[bold green]Output:[/bold green]")
    console.print(Syntax(example.output_content, "python"))

    if example.notes:
        console.print(f"\n[bold yellow]Notes:[/bold yellow]\n{example.notes}")


@examples.command("add")
@click.option("--name", "-n", required=True, help="Example name")
@click.option("--type", "-t", "comp_type", required=True, help="Component type")
@click.option("--tags", multiple=True, help="Tags for the example")
@click.option("--input-file", type=click.Path(exists=True), help="File containing input code")
@click.option("--output-file", type=click.Path(exists=True), help="File containing output code")
@click.pass_context
def examples_add(ctx, name: str, comp_type: str, tags: tuple, input_file: str, output_file: str):
    """Add a new conversion example from files."""
    config_path = get_config_path(ctx)
    examples_dir = config_path / "examples" / "custom"
    examples_dir.mkdir(parents=True, exist_ok=True)

    input_content = ""
    output_content = ""

    if input_file:
        with open(input_file) as f:
            input_content = f.read()

    if output_file:
        with open(output_file) as f:
            output_content = f.read()

    example_data = {
        "name": name,
        "description": f"Custom example for {comp_type}",
        "tags": list(tags) if tags else [comp_type.lower()],
        "input": {
            "type": "ssis_component",
            "component_type": comp_type,
            "content": input_content
        },
        "output": {
            "type": "pyspark",
            "content": output_content
        },
        "notes": "User-provided custom example"
    }

    example_file = examples_dir / f"{name}.yaml"
    with open(example_file, "w") as f:
        yaml.dump(example_data, f, default_flow_style=False)

    console.print(f"[green]✓[/green] Created example: {example_file}")
    console.print("  Edit the file to add description and notes")


# =============================================================================
# ANALYSIS COMMANDS
# =============================================================================

@cli.command()
@click.option(
    "--source", "-s",
    type=click.Path(exists=True),
    required=True,
    help="Path to SSIS .dtsx file or SQL Server .sql file"
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format"
)
@click.pass_context
def analyze(ctx, source: str, format: str):
    """Analyze a source file without converting.

    Parses the source and displays structure, complexity, and conversion notes.
    """
    source_path = Path(source)

    if source_path.suffix.lower() == ".dtsx":
        _analyze_ssis(source_path, format, ctx)
    elif source_path.suffix.lower() == ".sql":
        _analyze_sql(source_path, format)
    else:
        console.print(f"[red]Unsupported file type: {source_path.suffix}[/red]")
        raise click.Abort()


def _analyze_ssis(path: Path, format: str, ctx):
    """Analyze an SSIS package."""
    from src.ssis.dtsx_parser import DTSXParser
    from src.config_loader import ConfigLoader, ExampleMatcher

    console.print(f"\n[bold blue]Analyzing SSIS Package:[/bold blue] {path.name}\n")

    parser = DTSXParser(path)
    package = parser.parse()
    summary = parser.get_summary()

    if format == "json":
        console.print(Syntax(json.dumps(summary, indent=2), "json"))
    elif format == "yaml":
        console.print(Syntax(yaml.dump(summary, default_flow_style=False), "yaml"))
    else:
        _display_ssis_analysis(package, summary, ctx)


def _display_ssis_analysis(package, summary, ctx):
    """Display SSIS analysis in rich table format."""
    # Package overview
    console.print(Panel(
        f"[bold]{package.name}[/bold]\n{package.description}",
        title="Package Overview"
    ))

    # Summary table
    summary_table = Table(title="Package Summary")
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Count", justify="right", style="green")

    summary_table.add_row("Connections", str(summary["connections"]))
    summary_table.add_row("Variables", str(summary["variables"]))
    summary_table.add_row("Control Flow Tasks", str(summary["control_flow_tasks"]))
    summary_table.add_row("Data Flows", str(summary["data_flows"]))
    summary_table.add_row("Event Handlers", str(summary["event_handlers"]))

    console.print(summary_table)
    console.print()

    # Load config and examples for mapping info
    try:
        from src.config_loader import ConfigLoader
        loader = ConfigLoader(get_config_path(ctx))
        config = loader.load_config()
        example_matcher = ExampleMatcher(config.examples)
        has_config = True
    except:
        has_config = False

    # Connections table
    if package.connections:
        conn_table = Table(title="Connection Managers")
        conn_table.add_column("Name", style="cyan")
        conn_table.add_column("Type", style="yellow")
        conn_table.add_column("Server", style="green")
        conn_table.add_column("Database", style="green")

        for conn in package.connections:
            conn_table.add_row(
                conn.name,
                conn.connection_type,
                conn.server or "-",
                conn.database or "-"
            )

        console.print(conn_table)
        console.print()

    # Variables table
    if package.variables:
        var_table = Table(title="Package Variables")
        var_table.add_column("Name", style="cyan")
        var_table.add_column("Type", style="yellow")
        var_table.add_column("Value/Expression", style="green", max_width=50)

        for var in package.variables:
            value = var.expression if var.is_expression else str(var.value or "")
            if len(value) > 47:
                value = value[:47] + "..."
            var_table.add_row(
                f"{var.namespace}::{var.name}",
                var.data_type,
                value
            )

        console.print(var_table)
        console.print()

    # Control flow tasks with example availability
    if package.tasks:
        task_table = Table(title="Control Flow Tasks")
        task_table.add_column("Name", style="cyan")
        task_table.add_column("Type", style="yellow")
        task_table.add_column("Databricks Target", style="green")
        if has_config:
            task_table.add_column("Examples", justify="center", style="magenta")

        mapping_file = Path(__file__).parent.parent / "config" / "ssis_component_mappings.yaml"
        mappings = {}
        if mapping_file.exists():
            with open(mapping_file) as f:
                mappings = yaml.safe_load(f).get("control_flow_tasks", {})

        for task in package.tasks:
            target_info = mappings.get(task.task_type, {})
            target = target_info.get("target", "unknown")
            confidence = target_info.get("confidence", "low")
            confidence_color = {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "white")

            row = [
                task.name,
                task.task_type,
                f"[{confidence_color}]{target}[/{confidence_color}]"
            ]

            if has_config:
                examples = example_matcher.find_matches(component_type=task.task_type, max_results=1)
                row.append("✓" if examples else "-")

            task_table.add_row(*row)

            for child in task.child_tasks:
                child_target = mappings.get(child.task_type, {}).get("target", "unknown")
                child_row = [f"  └─ {child.name}", child.task_type, child_target]
                if has_config:
                    examples = example_matcher.find_matches(component_type=child.task_type, max_results=1)
                    child_row.append("✓" if examples else "-")
                task_table.add_row(*child_row)

        console.print(task_table)
        console.print()

    # Data flow components
    _display_data_flows(package.tasks, has_config, example_matcher if has_config else None)


def _display_data_flows(tasks, has_config, example_matcher):
    """Display data flow components."""
    for task in tasks:
        if task.data_flow and task.data_flow.components:
            df_table = Table(title=f"Data Flow: {task.data_flow.name}")
            df_table.add_column("Component", style="cyan")
            df_table.add_column("Type", style="yellow")
            df_table.add_column("Outputs", justify="right", style="green")
            if has_config:
                df_table.add_column("Example", justify="center", style="magenta")

            for comp in task.data_flow.components:
                row = [comp.name, comp.component_type, str(len(comp.output_columns))]
                if has_config:
                    examples = example_matcher.find_matches(component_type=comp.component_type, max_results=1)
                    row.append("✓" if examples else "-")
                df_table.add_row(*row)

            console.print(df_table)
            console.print()

        _display_data_flows(task.child_tasks, has_config, example_matcher)


def _analyze_sql(path: Path, format: str):
    """Analyze a SQL Server file."""
    console.print(f"[yellow]SQL analysis not yet implemented[/yellow]")
    console.print(f"File: {path}")


# =============================================================================
# CONVERSION COMMANDS
# =============================================================================

@cli.command("convert-ssis")
@click.option("--source", "-s", type=click.Path(exists=True), required=True, help="Path to SSIS .dtsx file")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--target", "-t", type=click.Choice(["workflow", "notebooks", "dlt"]), default="workflow", help="Target format")
@click.option("--min-score", type=float, default=0.70, help="Minimum quality score")
@click.option("--dry-run", is_flag=True, help="Parse without generating output")
@click.option("--show-prompts", is_flag=True, help="Display conversion prompts (for debugging)")
@click.pass_context
def convert_ssis(ctx, source: str, output: str, target: str, min_score: float, dry_run: bool, show_prompts: bool):
    """Convert an SSIS package to Databricks.

    Generates Databricks workflow definitions and/or PySpark notebooks
    from SSIS package files.
    """
    from src.ssis.dtsx_parser import DTSXParser
    from src.conversion.ssis_to_workflow import SSISToWorkflowConverter
    from src.conversion.ssis_to_notebook import SSISToNotebookConverter
    from src.config_loader import ConfigLoader

    source_path = Path(source)
    output_path = Path(output)

    # Load configuration
    loader = ConfigLoader(get_config_path(ctx))
    config = loader.load_config()

    # Apply overrides
    if ctx.obj.get("catalog_override"):
        config.default_catalog = ctx.obj["catalog_override"]
    if ctx.obj.get("schema_override"):
        config.default_schema = ctx.obj["schema_override"]

    console.print(f"\n[bold blue]Converting SSIS Package:[/bold blue] {source_path.name}")
    console.print(f"[bold blue]Target:[/bold blue] {target}")
    console.print(f"[bold blue]Output:[/bold blue] {output_path}")
    console.print(f"[bold blue]Catalog:[/bold blue] {config.default_catalog}.{config.default_schema}\n")

    # Parse package
    with console.status("[bold green]Parsing SSIS package..."):
        parser = DTSXParser(source_path)
        package = parser.parse()

    console.print(f"[green]✓[/green] Parsed {len(package.tasks)} tasks, {len(package.connections)} connections")

    # Show prompts if requested
    if show_prompts:
        from src.conversion.prompt_engine import PromptEngine
        engine = PromptEngine(config)
        prompt = engine.build_workflow_conversion_prompt(package)
        console.print("\n[bold yellow]Conversion Prompt:[/bold yellow]")
        console.print(Syntax(engine.get_full_prompt_text(prompt)[:2000] + "...", "markdown"))

    if dry_run:
        console.print("\n[yellow]Dry run - no files will be generated[/yellow]")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    package_output = output_path / package.name
    package_output.mkdir(exist_ok=True)

    # Convert
    if target in ("workflow", "notebooks"):
        with console.status("[bold green]Generating Databricks workflow..."):
            workflow_converter = SSISToWorkflowConverter()
            workflow = workflow_converter.convert(package)
            workflow_json = workflow_converter.to_json(workflow)

        workflow_file = package_output / "workflow.json"
        with open(workflow_file, "w") as f:
            f.write(workflow_json)
        console.print(f"[green]✓[/green] Generated {workflow_file}")

        notebooks_dir = package_output / "notebooks"
        notebooks_dir.mkdir(exist_ok=True)

        with console.status("[bold green]Generating PySpark notebooks..."):
            notebook_converter = SSISToNotebookConverter()
            for task in package.tasks:
                _generate_notebooks_for_task(task, notebook_converter, notebooks_dir)

        console.print(f"[green]✓[/green] Generated notebooks in {notebooks_dir}")

    elif target == "dlt":
        console.print("[yellow]DLT conversion not yet implemented[/yellow]")

    console.print(Panel(
        f"[green]Conversion complete![/green]\n\n"
        f"Output directory: {package_output}\n"
        f"- workflow.json: Databricks Jobs API definition\n"
        f"- notebooks/: PySpark notebooks for data flows",
        title="Summary"
    ))


def _generate_notebooks_for_task(task, converter, output_dir: Path):
    """Recursively generate notebooks for tasks with data flows."""
    if task.data_flow and task.data_flow.components:
        notebook_code = converter.convert_data_flow(task.data_flow, task.name)
        notebook_file = output_dir / f"{_sanitize_name(task.name)}.py"
        with open(notebook_file, "w") as f:
            f.write(notebook_code)

    for child in task.child_tasks:
        _generate_notebooks_for_task(child, converter, output_dir)


def _sanitize_name(name: str) -> str:
    """Convert task name to valid filename."""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")


@cli.command()
@click.option("--source", "-s", type=click.Path(exists=True), required=True, help="SQL Server .sql file")
@click.option("--output", "-o", type=click.Path(), required=True, help="Output directory")
@click.option("--target", "-t", type=click.Choice(["spark", "dbt"]), default="spark", help="Target format")
def convert(source: str, output: str, target: str):
    """Convert a SQL Server stored procedure to Databricks."""
    console.print("[yellow]SQL conversion not yet implemented[/yellow]")
    console.print(f"Source: {source}")
    console.print(f"Output: {output}")
    console.print(f"Target: {target}")


# =============================================================================
# UTILITY COMMANDS
# =============================================================================

@cli.command()
@click.argument("output_dir", type=click.Path(), default="samples/test_data")
def generate_test_data(output_dir: str):
    """Generate synthetic test data for validation."""
    import subprocess
    import sys

    script_path = Path(__file__).parent.parent / "scripts" / "generate_test_data.py"
    console.print(f"[bold blue]Generating test data...[/bold blue]")
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    if result.returncode == 0:
        console.print(result.stdout)
    else:
        console.print(f"[red]Error:[/red] {result.stderr}")


@cli.command()
@click.pass_context
def validate(ctx):
    """Run validation against test data."""
    import subprocess
    import sys

    script_path = Path(__file__).parent.parent / "scripts" / "run_and_validate.py"
    console.print(f"[bold blue]Running validation...[/bold blue]")
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    console.print(result.stdout)
    if result.stderr:
        console.print(result.stderr)


if __name__ == "__main__":
    cli()
