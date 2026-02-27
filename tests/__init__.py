"""Test suite for Conversion Control Tower."""


def find_task_recursive(tasks, name):
    """Helper: find a task by name in a nested task hierarchy."""
    for task in tasks:
        if task.name == name:
            return task
        found = find_task_recursive(task.child_tasks, name)
        if found:
            return found
    return None


def find_component_by_type(package, component_type):
    """Helper: find all data flow components of a given type across all tasks."""
    components = []
    _collect_components(package.tasks, component_type, components)
    return components


def _collect_components(tasks, component_type, results):
    for task in tasks:
        if task.data_flow:
            for comp in task.data_flow.components:
                if comp.component_type == component_type:
                    results.append(comp)
        _collect_components(task.child_tasks, component_type, results)
