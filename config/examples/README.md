# Conversion Examples

This directory contains input/output examples that guide the conversion process.
Each example shows how a specific SSIS pattern should be converted to Databricks.

## Directory Structure

```
examples/
├── sources/          # OLE DB Source, Flat File Source examples
├── lookups/          # Lookup transformation examples
├── transforms/       # Derived Column, Conditional Split, etc.
├── destinations/     # OLE DB Destination, Flat File Destination
├── scd/              # Slowly Changing Dimension patterns
└── workflows/        # Control flow and workflow examples
```

## Example File Format

Each `.yaml` file contains:
- `name`: Descriptive name for the pattern
- `description`: When to use this pattern
- `input`: The SSIS code/XML/expression
- `output`: The expected Databricks code
- `notes`: Additional context or warnings

## How Examples Are Used

1. **Pattern Matching**: The converter matches input patterns to find relevant examples
2. **Prompt Enhancement**: Examples are included in Claude API prompts for context
3. **Validation**: Generated code is compared against example patterns
4. **Learning**: Add new examples to improve conversion quality over time

## Adding Custom Examples

Create a new `.yaml` file in the appropriate directory:

```yaml
name: my_custom_pattern
description: When to use this pattern
tags: [lookup, dimension, broadcast]

input:
  type: ssis_component  # or sql, expression
  content: |
    <your SSIS XML or SQL here>

output:
  type: pyspark  # or sql, dlt
  content: |
    # Your expected Databricks code here

notes: |
  Any special considerations or warnings
```
