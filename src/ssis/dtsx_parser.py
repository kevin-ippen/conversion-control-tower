"""
SSIS .dtsx Package Parser

Parses SSIS package XML files into structured Python objects for conversion.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import re


@dataclass
class SSISConnection:
    """Represents an SSIS Connection Manager."""
    name: str
    connection_type: str
    connection_string: str
    dtsid: str
    server: Optional[str] = None
    database: Optional[str] = None
    property_expressions: Dict[str, str] = field(default_factory=dict)


@dataclass
class SSISVariable:
    """Represents an SSIS Package Variable."""
    name: str
    namespace: str
    data_type: str
    value: Any
    expression: Optional[str] = None
    is_expression: bool = False
    dtsid: str = ""


@dataclass
class SSISDataFlowComponent:
    """Represents a component within an SSIS Data Flow Task."""
    name: str
    component_type: str
    ref_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    input_columns: List[Dict] = field(default_factory=list)
    output_columns: List[Dict] = field(default_factory=list)


@dataclass
class SSISDataFlowPath:
    """Represents a path between components in a Data Flow."""
    name: str
    source_ref: str
    dest_ref: str


@dataclass
class SSISDataFlow:
    """Represents the pipeline inside a Data Flow Task."""
    name: str
    components: List[SSISDataFlowComponent] = field(default_factory=list)
    paths: List[SSISDataFlowPath] = field(default_factory=list)


@dataclass
class SSISPrecedenceConstraint:
    """Represents a precedence constraint between tasks."""
    from_task: str
    to_task: str
    eval_op: Optional[str] = None
    expression: Optional[str] = None
    value: str = "0"  # 0=Success, 1=Failure, 2=Completion


@dataclass
class SSISTask:
    """Represents an SSIS Control Flow Task."""
    name: str
    task_type: str
    ref_id: str
    dtsid: str
    properties: Dict[str, Any] = field(default_factory=dict)
    data_flow: Optional[SSISDataFlow] = None
    child_tasks: List['SSISTask'] = field(default_factory=list)
    precedence_constraints: List[SSISPrecedenceConstraint] = field(default_factory=list)


@dataclass
class SSISEventHandler:
    """Represents an SSIS Event Handler."""
    event_name: str
    tasks: List[SSISTask] = field(default_factory=list)
    precedence_constraints: List[SSISPrecedenceConstraint] = field(default_factory=list)


@dataclass
class SSISPackageParameter:
    """Represents an SSIS Package Parameter (distinct from Variables)."""
    name: str
    data_type: str
    value: Any
    dtsid: str = ""


@dataclass
class SSISPackage:
    """Represents a complete SSIS Package."""
    name: str
    description: str
    connections: List[SSISConnection] = field(default_factory=list)
    variables: List[SSISVariable] = field(default_factory=list)
    package_parameters: List[SSISPackageParameter] = field(default_factory=list)
    tasks: List[SSISTask] = field(default_factory=list)
    precedence_constraints: List[SSISPrecedenceConstraint] = field(default_factory=list)
    event_handlers: List[SSISEventHandler] = field(default_factory=list)


class DTSXParser:
    """
    Parses SSIS .dtsx package files into structured Python objects.
    """

    # XML namespaces used in DTSX files
    NAMESPACES = {
        'DTS': 'www.microsoft.com/SqlServer/Dts',
        'SQLTask': 'www.microsoft.com/sqlserver/dts/tasks/sqltask',
        'SendMailTask': 'www.microsoft.com/sqlserver/dts/tasks/sendmailtask'
    }

    # Map of DTS:CreationName patterns to task types
    TASK_TYPE_MAP = {
        'Microsoft.ExecuteSQLTask': 'ExecuteSQLTask',
        'Microsoft.Pipeline': 'DataFlowTask',
        'Microsoft.ScriptTask': 'ScriptTask',
        'Microsoft.FileSystemTask': 'FileSystemTask',
        'Microsoft.SendMailTask': 'SendMailTask',
        'Microsoft.ExecutePackageTask': 'ExecutePackageTask',
        'Microsoft.FTPTask': 'FTPTask',
        'STOCK:SEQUENCE': 'SequenceContainer',
        'STOCK:FORLOOP': 'ForLoopContainer',
        'STOCK:FOREACHLOOP': 'ForEachLoopContainer',
    }

    # Map of component class IDs to component types
    COMPONENT_TYPE_MAP = {
        'Microsoft.OLEDBSource': 'OLEDBSource',
        'Microsoft.OLEDBDestination': 'OLEDBDestination',
        'Microsoft.Lookup': 'Lookup',
        'Microsoft.DerivedColumn': 'DerivedColumn',
        'Microsoft.ConditionalSplit': 'ConditionalSplit',
        'Microsoft.Aggregate': 'Aggregate',
        'Microsoft.Sort': 'Sort',
        'Microsoft.UnionAll': 'UnionAll',
        'Microsoft.Multicast': 'Multicast',
        'Microsoft.RowCount': 'RowCount',
        'Microsoft.FlatFileDestination': 'FlatFileDestination',
        'Microsoft.FlatFileSource': 'FlatFileSource',
        'Microsoft.SlowlyChangingDimension': 'SlowlyChangingDimension',
        'Microsoft.OLEDBCommand': 'OLEDBCommand',
        'Microsoft.MergeJoin': 'MergeJoin',
        'Microsoft.DataConvert': 'DataConversion',
        'Microsoft.DataConversion': 'DataConversion',
    }

    def __init__(self, dtsx_path: Union[Path, str]):
        self.path = Path(dtsx_path)
        self.tree = ET.parse(self.path)
        self.root = self.tree.getroot()

    def parse(self) -> SSISPackage:
        """Main entry point - parse entire package."""
        return SSISPackage(
            name=self._get_package_name(),
            description=self._get_package_description(),
            connections=self._parse_connections(),
            variables=self._parse_variables(),
            package_parameters=self._parse_package_parameters(),
            tasks=self._parse_control_flow(),
            precedence_constraints=self._parse_package_precedence_constraints(),
            event_handlers=self._parse_event_handlers()
        )

    def _get_package_name(self) -> str:
        """Get the package name from root element."""
        return self.root.get(f'{{{self.NAMESPACES["DTS"]}}}ObjectName', self.path.stem)

    def _get_package_description(self) -> str:
        """Get the package description from root element."""
        return self.root.get(f'{{{self.NAMESPACES["DTS"]}}}Description', '')

    def _parse_connections(self) -> List[SSISConnection]:
        """Extract all connection managers from the package."""
        connections = []

        # Find ConnectionManagers element
        conn_managers_elem = self.root.find('DTS:ConnectionManagers', self.NAMESPACES)
        if conn_managers_elem is None:
            return connections

        for cm in conn_managers_elem.findall('DTS:ConnectionManager', self.NAMESPACES):
            name = cm.get(f'{{{self.NAMESPACES["DTS"]}}}ObjectName', '')
            creation_name = cm.get(f'{{{self.NAMESPACES["DTS"]}}}CreationName', '')
            dtsid = cm.get(f'{{{self.NAMESPACES["DTS"]}}}DTSID', '')

            # Parse PropertyExpressions (e.g. ConnectionString from variable)
            prop_expressions = {}
            for pe in cm.findall('DTS:PropertyExpression', self.NAMESPACES):
                pe_name = pe.get(f'{{{self.NAMESPACES["DTS"]}}}Name', '')
                if pe_name and pe.text:
                    prop_expressions[pe_name] = pe.text

            # Get connection string from nested ObjectData
            conn_string = ''
            obj_data = cm.find('DTS:ObjectData', self.NAMESPACES)
            if obj_data is not None:
                # Standard OLEDB/ADO.NET connections
                inner_cm = obj_data.find('DTS:ConnectionManager', self.NAMESPACES)
                if inner_cm is not None:
                    conn_string = inner_cm.get(
                        f'{{{self.NAMESPACES["DTS"]}}}ConnectionString',
                        inner_cm.get('DTS:ConnectionString', '')
                    )
                else:
                    # SMTP and other non-standard connection managers
                    smtp_cm = obj_data.find('SmtpConnectionManager')
                    if smtp_cm is not None:
                        conn_string = smtp_cm.get('ConnectionString', '')

            # Parse server/database from connection string
            server, database = self._parse_connection_string(conn_string)

            connections.append(SSISConnection(
                name=name,
                connection_type=creation_name,
                connection_string=conn_string,
                dtsid=dtsid,
                server=server,
                database=database,
                property_expressions=prop_expressions,
            ))

        return connections

    def _parse_connection_string(self, conn_string: str) -> tuple[Optional[str], Optional[str]]:
        """Extract server and database from a connection string."""
        server = None
        database = None

        if not conn_string:
            return server, database

        # Pattern for "Data Source=XXX"
        ds_match = re.search(r'Data Source=([^;]+)', conn_string, re.IGNORECASE)
        if ds_match:
            server = ds_match.group(1)

        # Pattern for "Initial Catalog=XXX"
        db_match = re.search(r'Initial Catalog=([^;]+)', conn_string, re.IGNORECASE)
        if db_match:
            database = db_match.group(1)

        return server, database

    def _parse_variables(self) -> List[SSISVariable]:
        """Extract all package variables."""
        variables = []

        vars_elem = self.root.find('DTS:Variables', self.NAMESPACES)
        if vars_elem is None:
            return variables

        for var in vars_elem.findall('DTS:Variable', self.NAMESPACES):
            name = var.get(f'{{{self.NAMESPACES["DTS"]}}}ObjectName', '')
            namespace = var.get(f'{{{self.NAMESPACES["DTS"]}}}Namespace', 'User')
            dtsid = var.get(f'{{{self.NAMESPACES["DTS"]}}}DTSID', '')
            is_expression = var.get(f'{{{self.NAMESPACES["DTS"]}}}EvaluateAsExpression', '') == 'True'
            expression = var.get(f'{{{self.NAMESPACES["DTS"]}}}Expression', None)

            # Get value from VariableValue element
            value_elem = var.find('DTS:VariableValue', self.NAMESPACES)
            data_type = ''
            value = None
            if value_elem is not None:
                data_type = value_elem.get(f'{{{self.NAMESPACES["DTS"]}}}DataType', '')
                value = value_elem.text

            variables.append(SSISVariable(
                name=name,
                namespace=namespace,
                data_type=data_type,
                value=value,
                expression=expression,
                is_expression=is_expression,
                dtsid=dtsid
            ))

        return variables

    def _parse_package_parameters(self) -> List[SSISPackageParameter]:
        """Extract package parameters (distinct from variables).

        Package parameters are defined in <DTS:PackageParameters> and represent
        inputs to the package (similar to stored procedure parameters).
        """
        parameters = []

        params_elem = self.root.find('DTS:PackageParameters', self.NAMESPACES)
        if params_elem is None:
            return parameters

        for param in params_elem.findall('DTS:PackageParameter', self.NAMESPACES):
            name = param.get(f'{{{self.NAMESPACES["DTS"]}}}ObjectName', '')
            data_type = param.get(f'{{{self.NAMESPACES["DTS"]}}}DataType', '')
            dtsid = param.get(f'{{{self.NAMESPACES["DTS"]}}}DTSID', '')

            # Get value from nested Property element
            value = None
            for prop in param.findall('DTS:Property', self.NAMESPACES):
                prop_name = prop.get(f'{{{self.NAMESPACES["DTS"]}}}Name', '')
                if prop_name == 'ParameterValue':
                    value = prop.text

            parameters.append(SSISPackageParameter(
                name=name,
                data_type=data_type,
                value=value,
                dtsid=dtsid,
            ))

        return parameters

    def _parse_control_flow(self) -> List[SSISTask]:
        """Parse the control flow task hierarchy."""
        tasks = []

        executables_elem = self.root.find('DTS:Executables', self.NAMESPACES)
        if executables_elem is None:
            return tasks

        for exe in executables_elem.findall('DTS:Executable', self.NAMESPACES):
            task = self._parse_task(exe)
            if task:
                tasks.append(task)

        return tasks

    def _parse_task(self, task_elem) -> Optional[SSISTask]:
        """Parse a single task element."""
        name = task_elem.get(f'{{{self.NAMESPACES["DTS"]}}}ObjectName', '')
        creation_name = task_elem.get(f'{{{self.NAMESPACES["DTS"]}}}CreationName', '')
        ref_id = task_elem.get(f'{{{self.NAMESPACES["DTS"]}}}refId', '')
        dtsid = task_elem.get(f'{{{self.NAMESPACES["DTS"]}}}DTSID', '')

        # Determine task type
        task_type = self._determine_task_type(creation_name)

        # Extract task-specific properties
        properties = self._extract_task_properties(task_elem, task_type)

        # Create base task
        task = SSISTask(
            name=name,
            task_type=task_type,
            ref_id=ref_id,
            dtsid=dtsid,
            properties=properties
        )

        # If Data Flow Task, parse the pipeline
        if task_type == 'DataFlowTask':
            task.data_flow = self._parse_data_flow(task_elem, name)

        # If container, parse child tasks
        if task_type in ('SequenceContainer', 'ForLoopContainer', 'ForEachLoopContainer'):
            child_executables = task_elem.find('DTS:Executables', self.NAMESPACES)
            if child_executables is not None:
                for child_exe in child_executables.findall('DTS:Executable', self.NAMESPACES):
                    child_task = self._parse_task(child_exe)
                    if child_task:
                        task.child_tasks.append(child_task)

            # Parse precedence constraints within container
            pc_elem = task_elem.find('DTS:PrecedenceConstraints', self.NAMESPACES)
            if pc_elem is not None:
                for pc in pc_elem.findall('DTS:PrecedenceConstraint', self.NAMESPACES):
                    constraint = self._parse_precedence_constraint(pc)
                    if constraint:
                        task.precedence_constraints.append(constraint)

        return task

    def _determine_task_type(self, creation_name: str) -> str:
        """Map creation name to task type."""
        for pattern, task_type in self.TASK_TYPE_MAP.items():
            if pattern in creation_name:
                return task_type
        return 'UnknownTask'

    def _extract_task_properties(self, task_elem, task_type: str) -> Dict[str, Any]:
        """Extract task-specific properties based on type."""
        props = {}

        obj_data = task_elem.find('DTS:ObjectData', self.NAMESPACES)
        if obj_data is None:
            return props

        if task_type == 'ExecuteSQLTask':
            sql_task = obj_data.find('SQLTask:SqlTaskData', self.NAMESPACES)
            if sql_task is not None:
                props['sql_statement'] = sql_task.get(
                    f'{{{self.NAMESPACES["SQLTask"]}}}SqlStatementSource',
                    sql_task.get('SQLTask:SqlStatementSource', '')
                )
                props['connection'] = sql_task.get(
                    f'{{{self.NAMESPACES["SQLTask"]}}}Connection',
                    sql_task.get('SQLTask:Connection', '')
                )
                props['result_type'] = sql_task.get(
                    f'{{{self.NAMESPACES["SQLTask"]}}}ResultType',
                    sql_task.get('SQLTask:ResultType', '')
                )

                # Parse parameter bindings
                bindings = []
                for binding in sql_task.findall('SQLTask:ParameterBinding', self.NAMESPACES):
                    bindings.append({
                        'parameter_name': binding.get(f'{{{self.NAMESPACES["SQLTask"]}}}ParameterName',
                                                     binding.get('SQLTask:ParameterName', '')),
                        'variable_name': binding.get(f'{{{self.NAMESPACES["SQLTask"]}}}DtsVariableName',
                                                    binding.get('SQLTask:DtsVariableName', ''))
                    })
                props['parameter_bindings'] = bindings

                # Parse result bindings
                result_bindings = []
                for binding in sql_task.findall('SQLTask:ResultBinding', self.NAMESPACES):
                    result_bindings.append({
                        'result_name': binding.get(f'{{{self.NAMESPACES["SQLTask"]}}}ResultName',
                                                  binding.get('SQLTask:ResultName', '')),
                        'variable_name': binding.get(f'{{{self.NAMESPACES["SQLTask"]}}}DtsVariableName',
                                                    binding.get('SQLTask:DtsVariableName', ''))
                    })
                props['result_bindings'] = result_bindings

        elif task_type == 'SendMailTask':
            mail_task = obj_data.find('SendMailTask:SendMailTaskData', self.NAMESPACES)
            if mail_task is not None:
                props['smtp_server'] = mail_task.get(
                    f'{{{self.NAMESPACES["SendMailTask"]}}}SMTPServer',
                    mail_task.get('SendMailTask:SMTPServer', '')
                )
                props['from'] = mail_task.get(
                    f'{{{self.NAMESPACES["SendMailTask"]}}}From',
                    mail_task.get('SendMailTask:From', '')
                )
                props['to'] = mail_task.get(
                    f'{{{self.NAMESPACES["SendMailTask"]}}}To',
                    mail_task.get('SendMailTask:To', '')
                )
                props['subject'] = mail_task.get(
                    f'{{{self.NAMESPACES["SendMailTask"]}}}Subject',
                    mail_task.get('SendMailTask:Subject', '')
                )
                props['message'] = mail_task.get(
                    f'{{{self.NAMESPACES["SendMailTask"]}}}MessageSource',
                    mail_task.get('SendMailTask:MessageSource', '')
                )

        return props

    def _parse_data_flow(self, task_elem, task_name: str) -> SSISDataFlow:
        """Parse the pipeline inside a Data Flow Task."""
        data_flow = SSISDataFlow(name=task_name)

        obj_data = task_elem.find('DTS:ObjectData', self.NAMESPACES)
        if obj_data is None:
            return data_flow

        pipeline = obj_data.find('pipeline', None)  # No namespace for pipeline element
        if pipeline is None:
            return data_flow

        # Parse components
        components_elem = pipeline.find('components', None)
        if components_elem is not None:
            for comp in components_elem.findall('component', None):
                component = self._parse_data_flow_component(comp)
                if component:
                    data_flow.components.append(component)

        # Parse paths
        paths_elem = pipeline.find('paths', None)
        if paths_elem is not None:
            for path in paths_elem.findall('path', None):
                path_obj = SSISDataFlowPath(
                    name=path.get('name', ''),
                    source_ref=path.get('startId', ''),
                    dest_ref=path.get('endId', '')
                )
                data_flow.paths.append(path_obj)

        return data_flow

    def _parse_data_flow_component(self, comp_elem) -> Optional[SSISDataFlowComponent]:
        """Parse a single data flow component."""
        name = comp_elem.get('name', '')
        class_id = comp_elem.get('componentClassID', '')
        ref_id = comp_elem.get('refId', '')

        # Determine component type
        comp_type = self.COMPONENT_TYPE_MAP.get(class_id, class_id)

        # Extract component-level properties
        properties = {}
        props_elem = comp_elem.find('properties', None)
        if props_elem is not None:
            for prop in props_elem.findall('property', None):
                prop_name = prop.get('name', '')
                prop_value = prop.text or ''
                properties[prop_name] = prop_value

        # Extract component connections (which connection manager it uses)
        conns_elem = comp_elem.find('connections', None)
        if conns_elem is not None:
            for conn in conns_elem.findall('connection', None):
                conn_name = conn.get('name', '')
                conn_mgr_id = conn.get('connectionManagerID', '')
                conn_mgr_ref = conn.get('connectionManagerRefId', '')
                properties[f'_connection_{conn_name}'] = conn_mgr_ref or conn_mgr_id

        # Extract output columns with richer metadata
        output_columns = []
        outputs_elem = comp_elem.find('outputs', None)
        if outputs_elem is not None:
            for output in outputs_elem.findall('output', None):
                output_name = output.get('name', '')
                is_error_out = output.get('isErrorOut', '') == 'true'
                is_sorted = output.get('isSorted', '') == 'true'
                is_default_out = False
                exclusion_group = output.get('exclusionGroup', '')

                # Extract output-level properties (Conditional Split expressions, etc.)
                output_props = {}
                out_props_elem = output.find('properties', None)
                if out_props_elem is not None:
                    for prop in out_props_elem.findall('property', None):
                        prop_name = prop.get('name', '')
                        prop_value = prop.text or ''
                        output_props[prop_name] = prop_value
                        if prop_name == 'IsDefaultOut' and prop_value.lower() == 'true':
                            is_default_out = True

                # Store output-level metadata in properties for downstream use
                if output_props:
                    friendly_expr = output_props.get('FriendlyExpression', '')
                    eval_order = output_props.get('EvaluationOrder', '')
                    if friendly_expr or is_default_out:
                        properties[f'_output_{output_name}'] = {
                            'friendly_expression': friendly_expr,
                            'evaluation_order': eval_order,
                            'is_default_out': is_default_out,
                            'exclusion_group': exclusion_group,
                        }

                # Skip error outputs for column extraction
                if is_error_out:
                    continue

                output_cols = output.find('outputColumns', None)
                if output_cols is not None:
                    for col in output_cols.findall('outputColumn', None):
                        col_data = {
                            'name': col.get('name', ''),
                            'data_type': col.get('dataType', ''),
                            'ref_id': col.get('refId', ''),
                            'lineage_id': col.get('lineageId', ''),
                            'expression': col.get('expression', None),
                            'length': col.get('length', None),
                            'precision': col.get('precision', None),
                            'scale': col.get('scale', None),
                            'code_page': col.get('codePage', None),
                            'sort_key_position': col.get('sortKeyPosition', None),
                            'output_name': output_name,
                        }
                        # Extract column-level properties (e.g. SourceInputColumnLineageID)
                        col_props = col.find('properties', None)
                        if col_props is not None:
                            for cp in col_props.findall('property', None):
                                cp_name = cp.get('name', '')
                                cp_value = cp.text or ''
                                col_data[f'prop_{cp_name}'] = cp_value
                        output_columns.append(col_data)

                # Extract externalMetadataColumns as fallback type info
                ext_meta = output.find('externalMetadataColumns', None)
                if ext_meta is not None and ext_meta.get('isUsed', '') == 'True':
                    for ext_col in ext_meta.findall('externalMetadataColumn', None):
                        ext_name = ext_col.get('name', '')
                        # Enrich matching output column with external metadata
                        for oc in output_columns:
                            if oc['name'] == ext_name:
                                if not oc.get('data_type') and ext_col.get('dataType'):
                                    oc['data_type'] = ext_col.get('dataType', '')
                                if not oc.get('length') and ext_col.get('length'):
                                    oc['length'] = ext_col.get('length')
                                if not oc.get('precision') and ext_col.get('precision'):
                                    oc['precision'] = ext_col.get('precision')
                                if not oc.get('scale') and ext_col.get('scale'):
                                    oc['scale'] = ext_col.get('scale')
                                break

        # Extract input columns with richer metadata
        input_columns = []
        inputs_elem = comp_elem.find('inputs', None)
        if inputs_elem is not None:
            for input_elem in inputs_elem.findall('input', None):
                input_name = input_elem.get('name', '')
                input_cols = input_elem.find('inputColumns', None)
                if input_cols is not None:
                    for col in input_cols.findall('inputColumn', None):
                        input_columns.append({
                            'name': col.get('cachedName', col.get('name', '')),
                            'lineage_id': col.get('lineageId', ''),
                            'ref_id': col.get('refId', ''),
                            'cached_data_type': col.get('cachedDataType', ''),
                            'cached_length': col.get('cachedLength', None),
                            'cached_sort_key_position': col.get('cachedSortKeyPosition', None),
                            'input_name': input_name,
                        })

        return SSISDataFlowComponent(
            name=name,
            component_type=comp_type,
            ref_id=ref_id,
            properties=properties,
            input_columns=input_columns,
            output_columns=output_columns
        )

    def _parse_package_precedence_constraints(self) -> List[SSISPrecedenceConstraint]:
        """Parse package-level precedence constraints."""
        constraints = []

        pc_elem = self.root.find('DTS:PrecedenceConstraints', self.NAMESPACES)
        if pc_elem is None:
            return constraints

        for pc in pc_elem.findall('DTS:PrecedenceConstraint', self.NAMESPACES):
            constraint = self._parse_precedence_constraint(pc)
            if constraint:
                constraints.append(constraint)

        return constraints

    def _parse_precedence_constraint(self, pc_elem) -> Optional[SSISPrecedenceConstraint]:
        """Parse a single precedence constraint."""
        from_task = pc_elem.get(f'{{{self.NAMESPACES["DTS"]}}}From', '')
        to_task = pc_elem.get(f'{{{self.NAMESPACES["DTS"]}}}To', '')
        eval_op = pc_elem.get(f'{{{self.NAMESPACES["DTS"]}}}EvalOp', None)
        value = pc_elem.get(f'{{{self.NAMESPACES["DTS"]}}}Value', '0')

        # Get expression from property if exists
        expression = None
        for prop in pc_elem.findall('DTS:PropertyExpression', self.NAMESPACES):
            if 'Expression' in prop.get(f'{{{self.NAMESPACES["DTS"]}}}Name', ''):
                expression = prop.text

        return SSISPrecedenceConstraint(
            from_task=from_task,
            to_task=to_task,
            eval_op=eval_op,
            expression=expression,
            value=value
        )

    def _parse_event_handlers(self) -> List[SSISEventHandler]:
        """Parse package event handlers."""
        handlers = []

        handlers_elem = self.root.find('DTS:EventHandlers', self.NAMESPACES)
        if handlers_elem is None:
            return handlers

        for handler in handlers_elem.findall('DTS:EventHandler', self.NAMESPACES):
            event_name = handler.get(f'{{{self.NAMESPACES["DTS"]}}}EventName', '')

            # Parse tasks within handler
            tasks = []
            executables = handler.find('DTS:Executables', self.NAMESPACES)
            if executables is not None:
                for exe in executables.findall('DTS:Executable', self.NAMESPACES):
                    task = self._parse_task(exe)
                    if task:
                        tasks.append(task)

            # Parse precedence constraints within handler
            constraints = []
            pc_elem = handler.find('DTS:PrecedenceConstraints', self.NAMESPACES)
            if pc_elem is not None:
                for pc in pc_elem.findall('DTS:PrecedenceConstraint', self.NAMESPACES):
                    constraint = self._parse_precedence_constraint(pc)
                    if constraint:
                        constraints.append(constraint)

            handlers.append(SSISEventHandler(
                event_name=event_name,
                tasks=tasks,
                precedence_constraints=constraints
            ))

        return handlers

    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary of the package without full parsing."""
        package = self.parse()

        # Count data flow tasks
        data_flow_count = sum(1 for t in package.tasks if t.task_type == 'DataFlowTask')

        # Count nested tasks in containers
        def count_nested_data_flows(tasks: List[SSISTask]) -> int:
            count = 0
            for t in tasks:
                if t.task_type == 'DataFlowTask':
                    count += 1
                if t.child_tasks:
                    count += count_nested_data_flows(t.child_tasks)
            return count

        total_data_flows = count_nested_data_flows(package.tasks)

        return {
            'name': package.name,
            'description': package.description,
            'connections': len(package.connections),
            'variables': len(package.variables),
            'package_parameters': len(package.package_parameters),
            'control_flow_tasks': len(package.tasks),
            'data_flows': total_data_flows,
            'event_handlers': len(package.event_handlers),
            'connection_names': [c.name for c in package.connections],
            'connection_types': [c.connection_type for c in package.connections],
            'variable_names': [f"{v.namespace}::{v.name}" for v in package.variables],
            'parameter_names': [p.name for p in package.package_parameters],
            'task_names': [t.name for t in package.tasks]
        }


if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python dtsx_parser.py <path_to_dtsx>")
        sys.exit(1)

    parser = DTSXParser(sys.argv[1])
    summary = parser.get_summary()
    print(json.dumps(summary, indent=2))
