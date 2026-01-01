#!/usr/bin/env python3
"""Generate API documentation pages for MkDocs.

This script is used by mkdocs-gen-files to automatically generate
API reference documentation from the flashlight source code.
"""

from pathlib import Path

import mkdocs_gen_files

# Modules to document
MODULES = [
    "flashlight",
    "flashlight.tensor",
    "flashlight.nn",
    "flashlight.nn.module",
    "flashlight.nn.layers",
    "flashlight.nn.functional",
    "flashlight.optim",
    "flashlight.autograd",
    "flashlight.ops",
]

nav = mkdocs_gen_files.Nav()

# Generate reference pages for each module
for module_path in sorted(Path("flashlight").rglob("*.py")):
    # Skip private modules and __pycache__
    if module_path.name.startswith("_") and module_path.name != "__init__.py":
        continue
    if "__pycache__" in str(module_path):
        continue

    # Convert path to module name
    module_parts = list(module_path.with_suffix("").parts)

    # Handle __init__.py files
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
        if not module_parts:
            continue
        doc_path = Path("reference", *module_parts) / "index.md"
    else:
        doc_path = Path("reference", *module_parts).with_suffix(".md")

    module_name = ".".join(module_parts)

    # Skip if empty module name
    if not module_name:
        continue

    # Generate the documentation page
    with mkdocs_gen_files.open(doc_path, "w") as f:
        f.write(f"# {module_name}\n\n")
        f.write(f"::: {module_name}\n")
        f.write("    options:\n")
        f.write("      show_source: true\n")
        f.write("      show_bases: true\n")
        f.write("      members_order: source\n")

    # Set the edit path to the source file
    mkdocs_gen_files.set_edit_path(doc_path, module_path)

    # Add to navigation
    nav_parts = list(module_parts)
    if nav_parts:
        nav[tuple(nav_parts)] = str(doc_path)

# Generate the SUMMARY.md for literate-nav
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as f:
    f.write("# API Reference\n\n")
    f.writelines(nav.build_literate_nav())
