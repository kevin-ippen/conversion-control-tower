"""
Informatica PowerCenter XML Parser

Parses PowerCenter repository export XML files into structured Python objects
for LLM-agentic conversion to Databricks notebooks.

Follows the POWERMART XML schema:
  POWERMART > REPOSITORY > FOLDER > {SOURCE, TARGET, MAPPING, MAPPLET, SESSION, WORKFLOW}

Uses lxml.etree when available (XPath support, large-file robustness),
falls back to xml.etree.ElementTree.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

try:
    from lxml import etree as ET

    _USING_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET

    _USING_LXML = False
    logger.info("lxml not available, falling back to xml.etree.ElementTree")


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class InfaField:
    """A field (port) within a source, target, or transformation."""
    name: str
    datatype: str
    precision: int = 0
    scale: int = 0
    port_type: str = ""        # I, O, V, I/O, R (for ports)
    expression: Optional[str] = None
    default_value: Optional[str] = None
    is_groupby: bool = False   # Aggregator group-by flag
    key_type: Optional[str] = None   # PRIMARY, FOREIGN, etc.
    nullable: bool = True
    description: Optional[str] = None


@dataclass
class InfaSource:
    """A SOURCE definition (table/file)."""
    name: str
    database_type: str     # Oracle, SQL Server, Flat File, etc.
    db_name: str = ""
    owner_name: str = ""
    fields: List[InfaField] = field(default_factory=list)


@dataclass
class InfaTarget:
    """A TARGET definition."""
    name: str
    database_type: str
    db_name: str = ""
    owner_name: str = ""
    fields: List[InfaField] = field(default_factory=list)


@dataclass
class InfaTableAttribute:
    """A TABLEATTRIBUTE on a transformation (key config like join conditions)."""
    name: str
    value: str


@dataclass
class InfaTransformation:
    """A TRANSFORMATION within a mapping."""
    name: str
    type: str              # Source Qualifier, Expression, Filter, Joiner, Lookup, etc.
    template_type: Optional[str] = None
    description: str = ""
    fields: List[InfaField] = field(default_factory=list)
    table_attributes: List[InfaTableAttribute] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)

    # Type-specific convenience accessors
    @property
    def sql_query(self) -> Optional[str]:
        """Override SQL for Source Qualifier."""
        return self._attr("Sql Query")

    @property
    def source_filter(self) -> Optional[str]:
        return self._attr("Source Filter")

    @property
    def user_defined_join(self) -> Optional[str]:
        return self._attr("User Defined Join")

    @property
    def join_condition(self) -> Optional[str]:
        return self._attr("Join Condition")

    @property
    def join_type(self) -> Optional[str]:
        return self._attr("Join Type")

    @property
    def filter_condition(self) -> Optional[str]:
        return self._attr("FILTERCONDITION") or self._attr("Filter Condition")

    @property
    def lookup_table(self) -> Optional[str]:
        return self._attr("Lookup table name")

    @property
    def lookup_filter(self) -> Optional[str]:
        return self._attr("Lookup Source Filter")

    @property
    def lookup_policy(self) -> Optional[str]:
        return self._attr("Lookup policy on multiple match")

    @property
    def lookup_condition(self) -> Optional[str]:
        return self._attr("Lookup Condition")

    def _attr(self, name: str) -> Optional[str]:
        for ta in self.table_attributes:
            if ta.name == name:
                return ta.value
        return None


@dataclass
class InfaConnector:
    """A CONNECTOR wiring between transformation ports."""
    from_instance: str
    to_instance: str
    from_field: str
    to_field: str
    from_instance_type: str = ""
    to_instance_type: str = ""


@dataclass
class InfaMappingVariable:
    """A MAPPINGVARIABLE or MAPPING_PARAMETER."""
    name: str
    datatype: str
    initial_value: str = ""
    is_parameter: bool = False
    description: str = ""


@dataclass
class InfaMapping:
    """A MAPPING definition."""
    name: str
    description: str = ""
    sources: List[InfaSource] = field(default_factory=list)
    targets: List[InfaTarget] = field(default_factory=list)
    transformations: List[InfaTransformation] = field(default_factory=list)
    connectors: List[InfaConnector] = field(default_factory=list)
    variables: List[InfaMappingVariable] = field(default_factory=list)

    @property
    def complexity_score(self) -> float:
        """Heuristic complexity score (from POV doc)."""
        score = (
            len(self.transformations) * 1
            + sum(1 for t in self.transformations if t.type == "Lookup") * 2
            + sum(1 for t in self.transformations if t.type == "Router") * 2
            + sum(1 for t in self.transformations if t.type == "Rank") * 1.5
            + len(self.variables) * 0.5
        )
        # Count expression ports with nested IIF
        for t in self.transformations:
            if t.type == "Expression":
                for f in t.fields:
                    if f.expression and "IIF" in f.expression:
                        nesting = f.expression.count("IIF")
                        score += nesting * 1.5
        return score

    @property
    def complexity_level(self) -> str:
        s = self.complexity_score
        if s < 10:
            return "low"
        elif s < 30:
            return "medium"
        return "high"

    def topological_order(self) -> List[str]:
        """Return transformation names in topological order (sources → targets).

        Uses the connector graph to determine execution order.
        """
        # Build adjacency list from connectors
        graph: Dict[str, set] = {}
        all_nodes: set = set()
        for conn in self.connectors:
            all_nodes.add(conn.from_instance)
            all_nodes.add(conn.to_instance)
            if conn.to_instance not in graph:
                graph[conn.to_instance] = set()
            graph[conn.to_instance].add(conn.from_instance)

        # Also include transformations not in connectors
        for t in self.transformations:
            all_nodes.add(t.name)

        # Kahn's algorithm
        in_degree: Dict[str, int] = {n: 0 for n in all_nodes}
        adj: Dict[str, List[str]] = {n: [] for n in all_nodes}
        for conn in self.connectors:
            adj[conn.from_instance].append(conn.to_instance)
            in_degree[conn.to_instance] = in_degree.get(conn.to_instance, 0) + 1

        queue = [n for n in all_nodes if in_degree.get(n, 0) == 0]
        queue.sort()  # deterministic ordering
        result = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in sorted(adj.get(node, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result


@dataclass
class InfaMapplet:
    """A MAPPLET (reusable sub-graph, same structure as MAPPING)."""
    name: str
    description: str = ""
    transformations: List[InfaTransformation] = field(default_factory=list)
    connectors: List[InfaConnector] = field(default_factory=list)


@dataclass
class InfaSessionExtension:
    """Session-level reader/writer config."""
    name: str
    sub_type: str = ""
    instance_name: str = ""
    connection_name: str = ""
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class InfaSessionTransformOverride:
    """Per-transformation session override (SESSTRANSFORMATIONINST)."""
    transformation_name: str
    transformation_type: str = ""
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class InfaSession:
    """A SESSION definition."""
    name: str
    mapping_name: str
    description: str = ""
    extensions: List[InfaSessionExtension] = field(default_factory=list)
    transform_overrides: List[InfaSessionTransformOverride] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)  # pre/post SQL, commit interval, etc.


@dataclass
class InfaWorkflowTask:
    """A task within a WORKFLOW."""
    name: str
    type: str     # Session, Start, Decision, Email, Command, Timer
    instance_type: str = ""
    session_name: Optional[str] = None
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class InfaWorkflowLink:
    """A link (DAG edge) between workflow tasks."""
    from_task: str
    to_task: str
    condition: str = ""
    link_type: str = ""  # LINK_TYPE_DEFAULT, LINK_TYPE_DECISION, etc.


@dataclass
class InfaWorkflow:
    """A WORKFLOW definition."""
    name: str
    description: str = ""
    tasks: List[InfaWorkflowTask] = field(default_factory=list)
    links: List[InfaWorkflowLink] = field(default_factory=list)


@dataclass
class PowerCenterExport:
    """Top-level parsed PowerCenter repository export."""
    repository_name: str = ""
    folder_name: str = ""
    sources: List[InfaSource] = field(default_factory=list)
    targets: List[InfaTarget] = field(default_factory=list)
    mappings: List[InfaMapping] = field(default_factory=list)
    mapplets: List[InfaMapplet] = field(default_factory=list)
    sessions: List[InfaSession] = field(default_factory=list)
    workflows: List[InfaWorkflow] = field(default_factory=list)

    @property
    def mapping_count(self) -> int:
        return len(self.mappings)

    @property
    def total_transformations(self) -> int:
        return sum(len(m.transformations) for m in self.mappings)

    def get_mapping(self, name: str) -> Optional[InfaMapping]:
        for m in self.mappings:
            if m.name == name:
                return m
        return None


# ─── Parser ──────────────────────────────────────────────────────────────────

class PowerCenterParser:
    """Parses Informatica PowerCenter XML exports into structured objects.

    Accepts a single XML file or a directory of XML files. When given a
    directory, all files are merged into a unified PowerCenterExport.
    """

    # Transformation type abbreviations used in naming conventions
    TYPE_PREFIXES = {
        "sq_": "Source Qualifier",
        "exp_": "Expression",
        "fil_": "Filter",
        "jnr_": "Joiner",
        "lkp_": "Lookup",
        "rtr_": "Router",
        "agg_": "Aggregator",
        "rnk_": "Rank",
        "uni_": "Union",
        "nrm_": "Normalizer",
        "seq_": "Sequence Generator",
        "srt_": "Sorter",
        "m_": "Mapping",
        "map_": "Mapping",
    }

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def parse(self) -> PowerCenterExport:
        """Parse one or more XML files into a PowerCenterExport."""
        if self.path.is_dir():
            return self._parse_directory()
        return self._parse_file(self.path)

    def _parse_directory(self) -> PowerCenterExport:
        """Parse all XML files in a directory and merge."""
        merged = PowerCenterExport()
        xml_files = sorted(self.path.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {self.path}")

        for xml_file in xml_files:
            try:
                export = self._parse_file(xml_file)
                merged.sources.extend(export.sources)
                merged.targets.extend(export.targets)
                merged.mappings.extend(export.mappings)
                merged.mapplets.extend(export.mapplets)
                merged.sessions.extend(export.sessions)
                merged.workflows.extend(export.workflows)
                if not merged.repository_name:
                    merged.repository_name = export.repository_name
                    merged.folder_name = export.folder_name
            except Exception as e:
                logger.warning(f"Failed to parse {xml_file}: {e}")

        logger.info(
            f"Parsed {len(xml_files)} files: "
            f"{merged.mapping_count} mappings, {merged.total_transformations} transformations"
        )
        return merged

    def _parse_file(self, file_path: Path) -> PowerCenterExport:
        """Parse a single PowerCenter XML file."""
        tree = ET.parse(str(file_path))
        root = tree.getroot()

        # Strip namespace if present
        tag = root.tag
        if "}" in tag:
            tag = tag.split("}")[-1]

        if tag != "POWERMART":
            raise ValueError(f"Expected <POWERMART> root, got <{tag}>")

        export = PowerCenterExport()

        for repo in root.iter("REPOSITORY"):
            export.repository_name = repo.get("NAME", "")

            for folder in repo.iter("FOLDER"):
                export.folder_name = folder.get("NAME", "")

                # Parse top-level sources and targets
                for source_elem in folder.findall("SOURCE"):
                    export.sources.append(self._parse_source(source_elem))

                for target_elem in folder.findall("TARGET"):
                    export.targets.append(self._parse_target(target_elem))

                # Parse mapplets
                for mapplet_elem in folder.findall("MAPPLET"):
                    export.mapplets.append(self._parse_mapplet(mapplet_elem))

                # Parse mappings
                for mapping_elem in folder.findall("MAPPING"):
                    export.mappings.append(
                        self._parse_mapping(mapping_elem, export.sources, export.targets)
                    )

                # Parse sessions
                for session_elem in folder.findall("SESSION"):
                    export.sessions.append(self._parse_session(session_elem))

                # Parse workflows
                for wf_elem in folder.findall("WORKFLOW"):
                    export.workflows.append(self._parse_workflow(wf_elem))

        logger.info(
            f"Parsed {file_path.name}: {export.mapping_count} mappings, "
            f"{len(export.sessions)} sessions, {len(export.workflows)} workflows"
        )
        return export

    # ── Source / Target parsing ───────────────────────────────────────────

    def _parse_source(self, elem) -> InfaSource:
        fields = [self._parse_source_field(f) for f in elem.findall("SOURCEFIELD")]
        return InfaSource(
            name=elem.get("NAME", ""),
            database_type=elem.get("DATABASETYPE", ""),
            db_name=elem.get("DBDNAME", ""),
            owner_name=elem.get("OWNERNAME", ""),
            fields=fields,
        )

    def _parse_target(self, elem) -> InfaTarget:
        fields = [self._parse_target_field(f) for f in elem.findall("TARGETFIELD")]
        return InfaTarget(
            name=elem.get("NAME", ""),
            database_type=elem.get("DATABASETYPE", ""),
            db_name=elem.get("DBDNAME", ""),
            owner_name=elem.get("OWNERNAME", ""),
            fields=fields,
        )

    def _parse_source_field(self, elem) -> InfaField:
        return InfaField(
            name=elem.get("NAME", ""),
            datatype=elem.get("DATATYPE", ""),
            precision=int(elem.get("PRECISION", "0")),
            scale=int(elem.get("SCALE", "0")),
            key_type=elem.get("KEYTYPE", None),
            nullable=elem.get("NULLABLE", "NULL") == "NULL",
            description=elem.get("DESCRIPTION", ""),
        )

    def _parse_target_field(self, elem) -> InfaField:
        return InfaField(
            name=elem.get("NAME", ""),
            datatype=elem.get("DATATYPE", ""),
            precision=int(elem.get("PRECISION", "0")),
            scale=int(elem.get("SCALE", "0")),
            key_type=elem.get("KEYTYPE", None),
            nullable=elem.get("NULLABLE", "NULL") == "NULL",
            description=elem.get("DESCRIPTION", ""),
        )

    # ── Mapping parsing ──────────────────────────────────────────────────

    def _parse_mapping(
        self,
        elem,
        folder_sources: List[InfaSource],
        folder_targets: List[InfaTarget],
    ) -> InfaMapping:
        mapping = InfaMapping(
            name=elem.get("NAME", ""),
            description=elem.get("DESCRIPTION", ""),
        )

        # Inline sources/targets within the mapping
        for source_elem in elem.findall("SOURCE"):
            mapping.sources.append(self._parse_source(source_elem))
        for target_elem in elem.findall("TARGET"):
            mapping.targets.append(self._parse_target(target_elem))

        # If mapping has no inline sources, reference folder-level ones
        if not mapping.sources:
            mapping.sources = list(folder_sources)
        if not mapping.targets:
            mapping.targets = list(folder_targets)

        # Parse transformations
        for trans_elem in elem.findall("TRANSFORMATION"):
            mapping.transformations.append(self._parse_transformation(trans_elem))

        # Parse connectors
        for conn_elem in elem.findall("CONNECTOR"):
            mapping.connectors.append(self._parse_connector(conn_elem))

        # Parse mapping variables and parameters
        for var_elem in elem.findall("MAPPINGVARIABLE"):
            mapping.variables.append(self._parse_mapping_variable(var_elem, is_param=False))
        for param_elem in elem.findall("MAPPING_PARAMETER"):
            mapping.variables.append(self._parse_mapping_variable(param_elem, is_param=True))

        return mapping

    def _parse_mapplet(self, elem) -> InfaMapplet:
        mapplet = InfaMapplet(
            name=elem.get("NAME", ""),
            description=elem.get("DESCRIPTION", ""),
        )
        for trans_elem in elem.findall("TRANSFORMATION"):
            mapplet.transformations.append(self._parse_transformation(trans_elem))
        for conn_elem in elem.findall("CONNECTOR"):
            mapplet.connectors.append(self._parse_connector(conn_elem))
        return mapplet

    # ── Transformation parsing ───────────────────────────────────────────

    def _parse_transformation(self, elem) -> InfaTransformation:
        trans = InfaTransformation(
            name=elem.get("NAME", ""),
            type=elem.get("TYPE", ""),
            template_type=elem.get("TEMPLATETYPE", None),
            description=elem.get("DESCRIPTION", ""),
        )

        # Parse ports (TRANSFORMFIELD)
        for field_elem in elem.findall("TRANSFORMFIELD"):
            trans.fields.append(self._parse_transform_field(field_elem))

        # Parse table attributes (critical config per transformation type)
        for attr_elem in elem.findall("TABLEATTRIBUTE"):
            trans.table_attributes.append(InfaTableAttribute(
                name=attr_elem.get("NAME", ""),
                value=attr_elem.get("VALUE", ""),
            ))

        return trans

    def _parse_transform_field(self, elem) -> InfaField:
        port_type = elem.get("PORTTYPE", "")
        return InfaField(
            name=elem.get("NAME", ""),
            datatype=elem.get("DATATYPE", ""),
            precision=int(elem.get("PRECISION", "0")),
            scale=int(elem.get("SCALE", "0")),
            port_type=port_type,
            expression=elem.get("EXPRESSION", None),
            default_value=elem.get("DEFAULTVALUE", None),
            is_groupby=elem.get("GROUPBYPOSITION", "0") != "0",
            description=elem.get("DESCRIPTION", ""),
        )

    def _parse_connector(self, elem) -> InfaConnector:
        return InfaConnector(
            from_instance=elem.get("FROMINSTANCE", ""),
            to_instance=elem.get("TOINSTANCE", ""),
            from_field=elem.get("FROMFIELD", ""),
            to_field=elem.get("TOFIELD", ""),
            from_instance_type=elem.get("FROMINSTANCETYPE", ""),
            to_instance_type=elem.get("TOINSTANCETYPE", ""),
        )

    def _parse_mapping_variable(self, elem, is_param: bool) -> InfaMappingVariable:
        return InfaMappingVariable(
            name=elem.get("NAME", ""),
            datatype=elem.get("DATATYPE", ""),
            initial_value=elem.get("DEFAULTVALUE", elem.get("INITIALVALUE", "")),
            is_parameter=is_param,
            description=elem.get("DESCRIPTION", ""),
        )

    # ── Session parsing ──────────────────────────────────────────────────

    def _parse_session(self, elem) -> InfaSession:
        session = InfaSession(
            name=elem.get("NAME", ""),
            mapping_name=elem.get("MAPPINGNAME", ""),
            description=elem.get("DESCRIPTION", ""),
        )

        # Parse session extensions (reader/writer configs)
        for ext_elem in elem.findall("SESSIONEXTENSION"):
            session.extensions.append(InfaSessionExtension(
                name=ext_elem.get("NAME", ""),
                sub_type=ext_elem.get("SUBTYPE", ""),
                instance_name=ext_elem.get("SINSTANCENAME", ""),
                connection_name=ext_elem.get("DSQINSTNAME", ""),
                properties={
                    a.get("NAME", ""): a.get("VALUE", "")
                    for a in ext_elem.findall("ATTRIBUTE")
                },
            ))

        # Parse session-level attributes (pre/post SQL, commit interval, etc.)
        for attr_elem in elem.findall("ATTRIBUTE"):
            name = attr_elem.get("NAME", "")
            value = attr_elem.get("VALUE", "")
            if name:
                session.attributes[name] = value

        # Parse per-transformation session overrides
        for override_elem in elem.findall("SESSTRANSFORMATIONINST"):
            override = InfaSessionTransformOverride(
                transformation_name=override_elem.get("SINSTANCENAME", ""),
                transformation_type=override_elem.get("TRANSFORMATIONTYPE", ""),
                properties={
                    a.get("NAME", ""): a.get("VALUE", "")
                    for a in override_elem.findall("ATTRIBUTE")
                },
            )
            session.transform_overrides.append(override)

        return session

    # ── Workflow parsing ──────────────────────────────────────────────────

    def _parse_workflow(self, elem) -> InfaWorkflow:
        workflow = InfaWorkflow(
            name=elem.get("NAME", ""),
            description=elem.get("DESCRIPTION", ""),
        )

        for task_elem in elem.findall("TASK"):
            task = InfaWorkflowTask(
                name=task_elem.get("NAME", ""),
                type=task_elem.get("TYPE", ""),
                instance_type=task_elem.get("TASKINSTTYPE", ""),
                session_name=task_elem.get("SESSIONTASKNAME", None),
            )
            workflow.tasks.append(task)

        for link_elem in elem.findall("WORKFLOWLINK"):
            link = InfaWorkflowLink(
                from_task=link_elem.get("FROMTASK", ""),
                to_task=link_elem.get("TOTASK", ""),
                condition=link_elem.get("CONDITION", ""),
                link_type=link_elem.get("LINKTYPE", ""),
            )
            workflow.links.append(link)

        return workflow

    # ── Summary / inventory ──────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Quick inventory summary without deep analysis."""
        export = self.parse()
        return {
            "repository_name": export.repository_name,
            "folder_name": export.folder_name,
            "source_count": len(export.sources),
            "target_count": len(export.targets),
            "mapping_count": export.mapping_count,
            "mapplet_count": len(export.mapplets),
            "session_count": len(export.sessions),
            "workflow_count": len(export.workflows),
            "total_transformations": export.total_transformations,
            "mappings": [
                {
                    "name": m.name,
                    "transformations": len(m.transformations),
                    "connectors": len(m.connectors),
                    "variables": len(m.variables),
                    "complexity_score": m.complexity_score,
                    "complexity_level": m.complexity_level,
                    "transformation_types": list({t.type for t in m.transformations}),
                }
                for m in export.mappings
            ],
            "workflows": [
                {
                    "name": w.name,
                    "tasks": len(w.tasks),
                    "links": len(w.links),
                }
                for w in export.workflows
            ],
        }

    def build_structured_context(self, mapping_name: Optional[str] = None) -> str:
        """Build a structured text summary for LLM context injection.

        Instead of passing raw XML to the LLM, this provides a cleaned,
        structured representation that's easier for the model to reason about.
        """
        export = self.parse()
        mappings = [export.get_mapping(mapping_name)] if mapping_name else export.mappings
        mappings = [m for m in mappings if m is not None]

        parts = []
        parts.append(f"## PowerCenter Export: {export.repository_name}/{export.folder_name}")
        parts.append("")

        for mapping in mappings:
            parts.append(f"### Mapping: {mapping.name}")
            parts.append(f"Complexity: {mapping.complexity_level} (score: {mapping.complexity_score:.1f})")
            parts.append("")

            # Sources
            if mapping.sources:
                parts.append("**Sources:**")
                for src in mapping.sources:
                    cols = ", ".join(f"{f.name} {f.datatype}" for f in src.fields[:10])
                    parts.append(f"- {src.name} ({src.database_type}): [{cols}]")
                parts.append("")

            # Targets
            if mapping.targets:
                parts.append("**Targets:**")
                for tgt in mapping.targets:
                    cols = ", ".join(f"{f.name} {f.datatype}" for f in tgt.fields[:10])
                    parts.append(f"- {tgt.name} ({tgt.database_type}): [{cols}]")
                parts.append("")

            # Variables / Parameters
            if mapping.variables:
                parts.append("**Variables/Parameters:**")
                for v in mapping.variables:
                    kind = "PARAM" if v.is_parameter else "VAR"
                    parts.append(f"- $${v.name} ({kind}, {v.datatype}, default={v.initial_value!r})")
                parts.append("")

            # Transformations in topological order
            topo_order = mapping.topological_order()
            trans_by_name = {t.name: t for t in mapping.transformations}

            parts.append("**Transformations (execution order):**")
            for tname in topo_order:
                trans = trans_by_name.get(tname)
                if not trans:
                    continue
                parts.append(f"\n#### {trans.name} (Type: {trans.type})")

                # Key attributes
                if trans.sql_query:
                    parts.append(f"  SQL Override: {trans.sql_query}")
                if trans.source_filter:
                    parts.append(f"  Source Filter: {trans.source_filter}")
                if trans.join_condition:
                    parts.append(f"  Join: {trans.join_type or 'INNER'} ON {trans.join_condition}")
                if trans.filter_condition:
                    parts.append(f"  Filter: {trans.filter_condition}")
                if trans.lookup_table:
                    parts.append(f"  Lookup Table: {trans.lookup_table} (policy: {trans.lookup_policy})")
                    if trans.lookup_condition:
                        parts.append(f"  Lookup Condition: {trans.lookup_condition}")

                # Output ports with expressions
                output_ports = [f for f in trans.fields if "O" in f.port_type or f.port_type == ""]
                expr_ports = [f for f in trans.fields if f.expression]
                groupby_ports = [f for f in trans.fields if f.is_groupby]

                if groupby_ports:
                    parts.append(f"  GROUP BY: {', '.join(p.name for p in groupby_ports)}")

                if expr_ports:
                    parts.append("  Expressions:")
                    for p in expr_ports:
                        flag = f" [{p.port_type}]" if p.port_type else ""
                        parts.append(f"    {p.name}{flag} = {p.expression}")

            parts.append("")

            # Connector summary
            parts.append("**Data Flow (connectors):**")
            # Group connectors by from→to instance
            edges: Dict[Tuple[str, str], List[str]] = {}
            for c in mapping.connectors:
                key = (c.from_instance, c.to_instance)
                if key not in edges:
                    edges[key] = []
                edges[key].append(f"{c.from_field}→{c.to_field}")
            for (src, tgt), fields in edges.items():
                parts.append(f"  {src} → {tgt}: {len(fields)} field(s)")
            parts.append("")

        # Sessions
        if export.sessions:
            parts.append("### Sessions")
            for sess in export.sessions:
                parts.append(f"- {sess.name} → mapping: {sess.mapping_name}")
                if sess.attributes:
                    pre_sql = sess.attributes.get("Pre SQL", "")
                    post_sql = sess.attributes.get("Post SQL", "")
                    if pre_sql:
                        parts.append(f"  Pre-SQL: {pre_sql}")
                    if post_sql:
                        parts.append(f"  Post-SQL: {post_sql}")
                if sess.transform_overrides:
                    parts.append(f"  Session overrides: {len(sess.transform_overrides)} transformation(s)")
            parts.append("")

        # Workflows
        if export.workflows:
            parts.append("### Workflows")
            for wf in export.workflows:
                parts.append(f"- {wf.name}: {len(wf.tasks)} tasks, {len(wf.links)} links")
                for task in wf.tasks:
                    parts.append(f"  - {task.name} ({task.type})")
            parts.append("")

        return "\n".join(parts)


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python powercenter_parser.py <path_to_xml_or_dir>")
        sys.exit(1)

    parser = PowerCenterParser(sys.argv[1])
    summary = parser.get_summary()
    print(json.dumps(summary, indent=2))
