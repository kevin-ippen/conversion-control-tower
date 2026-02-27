"""
Configuration and Examples Loader

Loads conversion configuration, rules, and examples from YAML files.
Supports variable substitution and inheritance.
"""

import yaml
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ConversionExample:
    """Represents a single conversion example."""
    name: str
    description: str
    tags: List[str]
    input_type: str
    input_content: str
    output_type: str
    output_content: str
    notes: str = ""
    component_type: Optional[str] = None


@dataclass
class ConversionRule:
    """Represents a conversion rule."""
    name: str
    pattern: str
    replacement: str
    is_regex: bool = False
    notes: str = ""


@dataclass
class ConversionConfig:
    """Complete conversion configuration."""
    version: str
    config_name: str

    # Settings
    target_platform: str
    runtime_version: str
    language: str

    # Unity Catalog
    unity_catalog_enabled: bool
    default_catalog: str
    default_schema: str
    use_three_part_names: bool

    # Delta Lake
    delta_enabled: bool
    clustering_strategy: str

    # Secrets
    secrets_provider: str
    secrets_scope: str
    enforce_secrets: bool

    # Output preferences
    include_source_comments: bool
    add_todo_markers: bool
    generate_docs: bool

    # Rules
    sql_transforms: List[ConversionRule]
    data_type_mappings: Dict[str, str]
    naming_rules: Dict[str, Any]
    component_rules: Dict[str, Dict]
    error_handling: Dict[str, Any]
    performance_rules: Dict[str, Any]

    # Instructions
    global_instructions: str
    component_instructions: Dict[str, str]
    business_rules: str

    # Validation
    validation_rules: Dict[str, List[Dict]]
    quality_thresholds: Dict[str, float]

    # Variables for template substitution
    variables: Dict[str, str]

    # Examples (loaded separately)
    examples: List[ConversionExample] = field(default_factory=list)


class ConfigLoader:
    """
    Loads and manages conversion configuration.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize with optional custom config directory."""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)
        self.examples_dir = self.config_dir / "examples"

    def load_config(self, config_file: str = "conversion_config.yaml") -> ConversionConfig:
        """Load the main configuration file."""
        config_path = self.config_dir / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        # Parse configuration
        settings = raw_config.get("settings", {})
        target = settings.get("target", {})
        uc = settings.get("unity_catalog", {})
        delta = settings.get("delta", {})
        secrets = settings.get("secrets", {})
        output = settings.get("output", {})

        rules = raw_config.get("rules", {})
        instructions = raw_config.get("instructions", {})
        validation = raw_config.get("validation", {})

        # Build SQL transform rules
        sql_transforms = []
        for transform in rules.get("sql_dialect", {}).get("auto_transform", []):
            sql_transforms.append(ConversionRule(
                name=transform.get("from", ""),
                pattern=transform.get("pattern", transform.get("from", "")),
                replacement=transform.get("replacement", transform.get("to", "")),
                is_regex=transform.get("regex", False),
                notes=transform.get("notes", "")
            ))

        config = ConversionConfig(
            version=raw_config.get("version", "1.0"),
            config_name=raw_config.get("config_name", "default"),

            # Settings
            target_platform=target.get("platform", "databricks"),
            runtime_version=target.get("runtime_version", "14.3"),
            language=target.get("language", "python"),

            # Unity Catalog
            unity_catalog_enabled=uc.get("enabled", True),
            default_catalog=uc.get("default_catalog", "main"),
            default_schema=uc.get("default_schema", "default"),
            use_three_part_names=uc.get("use_three_part_names", True),

            # Delta
            delta_enabled=delta.get("enabled", True),
            clustering_strategy=delta.get("clustering_strategy", "liquid"),

            # Secrets
            secrets_provider=secrets.get("provider", "databricks"),
            secrets_scope=secrets.get("scope_name", "etl_secrets"),
            enforce_secrets=secrets.get("enforce_secrets", True),

            # Output
            include_source_comments=output.get("include_source_comments", True),
            add_todo_markers=output.get("add_todo_markers", True),
            generate_docs=output.get("generate_docs", True),

            # Rules
            sql_transforms=sql_transforms,
            data_type_mappings=rules.get("data_types", {}),
            naming_rules=rules.get("naming", {}),
            component_rules=rules.get("components", {}),
            error_handling=rules.get("error_handling", {}),
            performance_rules=rules.get("performance", {}),

            # Instructions
            global_instructions=instructions.get("global", ""),
            component_instructions=instructions.get("by_component", {}),
            business_rules=instructions.get("business_rules", ""),

            # Validation
            validation_rules={
                "pre_conversion": validation.get("pre_conversion", []),
                "post_conversion": validation.get("post_conversion", [])
            },
            quality_thresholds=validation.get("thresholds", {}).get("min_category_scores", {}),

            # Variables
            variables=raw_config.get("variables", {})
        )

        # Load examples
        config.examples = self.load_examples()

        return config

    def load_examples(self, tags_filter: Optional[List[str]] = None) -> List[ConversionExample]:
        """Load all examples from the examples directory."""
        examples = []

        if not self.examples_dir.exists():
            return examples

        # Walk through all subdirectories
        for yaml_file in self.examples_dir.rglob("*.yaml"):
            if yaml_file.name == "README.md":
                continue

            try:
                example = self._load_example_file(yaml_file)
                if example:
                    # Filter by tags if specified
                    if tags_filter:
                        if any(tag in example.tags for tag in tags_filter):
                            examples.append(example)
                    else:
                        examples.append(example)
            except Exception as e:
                print(f"Warning: Failed to load example {yaml_file}: {e}")

        return examples

    def _load_example_file(self, file_path: Path) -> Optional[ConversionExample]:
        """Load a single example file."""
        with open(file_path) as f:
            data = yaml.safe_load(f)

        if not data:
            return None

        input_data = data.get("input", {})
        output_data = data.get("output", {})

        return ConversionExample(
            name=data.get("name", file_path.stem),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            input_type=input_data.get("type", "unknown"),
            input_content=input_data.get("content", ""),
            output_type=output_data.get("type", "pyspark"),
            output_content=output_data.get("content", ""),
            notes=data.get("notes", ""),
            component_type=input_data.get("component_type")
        )

    def find_examples_for_component(self, component_type: str) -> List[ConversionExample]:
        """Find examples relevant to a specific component type."""
        matching = []
        component_lower = component_type.lower()

        for example in self.load_examples():
            # Match by component type
            if example.component_type and example.component_type.lower() == component_lower:
                matching.append(example)
            # Match by tags
            elif component_lower in [t.lower() for t in example.tags]:
                matching.append(example)

        return matching

    def substitute_variables(self, text: str, config: ConversionConfig, extra_vars: Optional[Dict] = None) -> str:
        """Substitute {{variable}} placeholders in text."""
        variables = dict(config.variables)
        if extra_vars:
            variables.update(extra_vars)

        # Also add config values as variables
        variables["catalog"] = config.default_catalog
        variables["schema"] = config.default_schema
        variables["secrets_scope"] = config.secrets_scope

        # Replace {{var}} patterns
        def replace_var(match):
            var_name = match.group(1)
            return variables.get(var_name, match.group(0))

        return re.sub(r"\{\{(\w+)\}\}", replace_var, text)


class ExampleMatcher:
    """
    Matches SSIS components to relevant conversion examples.
    """

    def __init__(self, examples: List[ConversionExample]):
        self.examples = examples
        self._build_index()

    def _build_index(self):
        """Build index for fast example lookup."""
        self.by_component_type = {}
        self.by_tag = {}

        for example in self.examples:
            # Index by component type
            if example.component_type:
                comp_type = example.component_type.lower()
                if comp_type not in self.by_component_type:
                    self.by_component_type[comp_type] = []
                self.by_component_type[comp_type].append(example)

            # Index by tags
            for tag in example.tags:
                tag_lower = tag.lower()
                if tag_lower not in self.by_tag:
                    self.by_tag[tag_lower] = []
                self.by_tag[tag_lower].append(example)

    def find_matches(
        self,
        component_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_results: int = 3
    ) -> List[ConversionExample]:
        """Find matching examples based on component type and/or tags."""
        candidates = set()

        # Match by component type
        if component_type:
            comp_lower = component_type.lower()
            if comp_lower in self.by_component_type:
                for ex in self.by_component_type[comp_lower]:
                    candidates.add(ex.name)

        # Match by tags
        if tags:
            for tag in tags:
                tag_lower = tag.lower()
                if tag_lower in self.by_tag:
                    for ex in self.by_tag[tag_lower]:
                        candidates.add(ex.name)

        # Get full examples
        results = [ex for ex in self.examples if ex.name in candidates]

        return results[:max_results]

    def get_best_example(
        self,
        component_type: str,
        context_tags: Optional[List[str]] = None
    ) -> Optional[ConversionExample]:
        """Get the single best matching example."""
        matches = self.find_matches(component_type, context_tags, max_results=1)
        return matches[0] if matches else None


if __name__ == "__main__":
    # Test loading
    loader = ConfigLoader()
    config = loader.load_config()

    print(f"Loaded config: {config.config_name}")
    print(f"  Target: {config.target_platform} {config.runtime_version}")
    print(f"  Unity Catalog: {config.default_catalog}.{config.default_schema}")
    print(f"  Examples loaded: {len(config.examples)}")

    for ex in config.examples:
        print(f"    - {ex.name}: {ex.description[:50]}...")
