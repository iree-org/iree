#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bundles JS companion modules for a wasm binary.

Reads a modules manifest (one "module:path" entry per line), optionally parses
the .wasm binary's import section for dead code elimination, transforms each
companion for inlining, and concatenates everything into a single .mjs file
with the entry point.

Each companion file must export exactly one function named createImports:

    export function createImports(context) {
        return {
            function_name(...args) { ... },
        };
    }

The bundler strips ESM import/export syntax, wraps each companion in an IIFE
that captures createImports, and generates a createWasmImports(context) function
that builds the full wasm import object.

The entry point (main.js) is appended last. Its relative imports (e.g.,
`import {Foo} from './foo.mjs'`) are recursively resolved and inlined into the
bundle so the output is fully self-contained. External ESM imports (e.g.,
'node:worker_threads') are preserved and hoisted by the JS engine.

Usage:
    python wasm_binary_bundler.py \\
        --wasm foo.wasm \\
        --main main.mjs \\
        --modules modules.manifest \\
        --output foo.mjs
"""

import argparse
import json
import os
import re
import sys


def decode_unsigned_leb128(data, offset):
    """Decodes an unsigned LEB128 integer from binary data."""
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, offset
        shift += 7
    raise ValueError("Truncated LEB128 at offset %d" % offset)


def skip_import_descriptor(data, offset):
    """Skips past a wasm import descriptor (kind + type details)."""
    kind = data[offset]
    offset += 1
    if kind == 0x00:
        # Function: type index (LEB128).
        _, offset = decode_unsigned_leb128(data, offset)
    elif kind == 0x01:
        # Table: elem_type (1 byte) + limits.
        offset += 1  # elem_type
        offset = _skip_limits(data, offset)
    elif kind == 0x02:
        # Memory: limits.
        offset = _skip_limits(data, offset)
    elif kind == 0x03:
        # Global: value_type (1 byte) + mutability (1 byte).
        offset += 2
    else:
        raise ValueError("Unknown import kind 0x%02x at offset %d" % (kind, offset - 1))
    return offset


def _skip_limits(data, offset):
    """Skips past a wasm limits encoding."""
    flags = data[offset]
    offset += 1
    _, offset = decode_unsigned_leb128(data, offset)  # min
    if flags & 0x01:
        _, offset = decode_unsigned_leb128(data, offset)  # max
    return offset


def parse_wasm_import_modules(wasm_path):
    """Parses a .wasm binary and returns the set of import module names."""
    with open(wasm_path, "rb") as f:
        data = f.read()

    # Validate magic and version.
    if len(data) < 8:
        raise ValueError("File too small to be a valid wasm binary")
    if data[:4] != b"\x00asm":
        raise ValueError("Not a wasm binary (bad magic)")
    if data[4:8] != b"\x01\x00\x00\x00":
        raise ValueError("Unsupported wasm version")

    offset = 8
    while offset < len(data):
        section_id = data[offset]
        offset += 1
        section_size, offset = decode_unsigned_leb128(data, offset)
        section_end = offset + section_size

        if section_id == 2:  # Import section.
            module_names = set()
            count, offset = decode_unsigned_leb128(data, offset)
            for _ in range(count):
                # Module name.
                name_length, offset = decode_unsigned_leb128(data, offset)
                module_name = data[offset : offset + name_length].decode("utf-8")
                offset += name_length
                module_names.add(module_name)

                # Field name (skip).
                field_length, offset = decode_unsigned_leb128(data, offset)
                offset += field_length

                # Import descriptor (skip).
                offset = skip_import_descriptor(data, offset)

            return module_names

        # Skip sections we don't care about.
        offset = section_end

    # No import section found — the binary has no imports.
    return set()


def transform_companion(source, module_name, source_path):
    """Transforms a JS companion module for inlining into the bundle.

    Strips ESM import/export syntax and wraps the content in an IIFE that
    captures the createImports function.

    Args:
        source: The JS source code as a string.
        module_name: The wasm module name (used for the variable name).
        source_path: Path to the source file (for diagnostics).

    Returns:
        Transformed JS code as a string.
    """
    lines = source.split("\n")
    output_lines = []
    found_create_imports = False

    for line in lines:
        stripped = line.lstrip()

        # Skip ESM import statements — companions must be self-contained.
        if stripped.startswith("import ") and " from " in stripped:
            continue

        # Strip 'export' keyword from function/const/class declarations.
        if stripped.startswith("export function "):
            line = line.replace("export function ", "function ", 1)
            if "createImports" in stripped:
                found_create_imports = True
        elif stripped.startswith("export default function "):
            line = line.replace("export default function ", "function ", 1)
        elif stripped.startswith("export const "):
            line = line.replace("export const ", "const ", 1)
        elif stripped.startswith("export class "):
            line = line.replace("export class ", "class ", 1)
        elif stripped.startswith("export {"):
            continue

        output_lines.append(line)

    if not found_create_imports:
        print(
            "WARNING: %s does not export createImports — module '%s' will "
            "return undefined at runtime" % (source_path, module_name),
            file=sys.stderr,
        )

    # Sanitize module name for use as a JS identifier.
    identifier = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)

    inner = "\n".join(output_lines)
    return (
        "// --- Module: %s (from %s) ---\n"
        "const _iree_mod_%s = (() => {\n"
        "%s\n"
        "return typeof createImports === 'function' ? createImports : undefined;\n"
        "})();\n"
    ) % (module_name, source_path, identifier, inner)


def generate_merger(module_names):
    """Generates the createWasmImports function that merges all companions."""
    entries = []
    for module_name in module_names:
        identifier = re.sub(r"[^a-zA-Z0-9_]", "_", module_name)
        entries.append('    "%s": _iree_mod_%s(context)' % (module_name, identifier))
    body = ",\n".join(entries)
    return (
        "\n"
        "// Creates the full wasm import object from all collected companions.\n"
        "// Each module's createImports(context) is called with the same context\n"
        "// object, which carries shared state (memory, ring buffers, etc.).\n"
        "export function createWasmImports(context) {\n"
        "  return {\n"
        "%s\n"
        "  };\n"
        "}\n"
    ) % body


def _is_relative_import(line):
    """Returns the relative path from an ESM import statement, or None."""
    stripped = line.lstrip()
    if not stripped.startswith("import ") or " from " not in stripped:
        return None
    # Extract the module specifier from: import ... from '...' or "...";
    match = re.search(r"""from\s+['"](\.[^'"]+)['"]""", stripped)
    if match:
        return match.group(1)
    return None


def _strip_exports(source):
    """Strips ESM export syntax from source, returning plain JS declarations."""
    lines = source.split("\n")
    output_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("export function "):
            line = line.replace("export function ", "function ", 1)
        elif stripped.startswith("export default function "):
            line = line.replace("export default function ", "function ", 1)
        elif stripped.startswith("export const "):
            line = line.replace("export const ", "const ", 1)
        elif stripped.startswith("export class "):
            line = line.replace("export class ", "class ", 1)
        elif stripped.startswith("export {"):
            continue
        output_lines.append(line)
    return "\n".join(output_lines)


def resolve_local_imports(source, source_path):
    """Resolves relative ESM imports recursively, producing self-contained JS.

    Scans source for `import ... from './...'` statements. For each relative
    import, reads the file, strips its exports, recursively resolves its own
    local imports, and prepends the result. The import statement is removed
    from the source. External imports (node builtins, bare specifiers) are
    preserved.

    Args:
        source: JS source code.
        source_path: Absolute path to the source file (for resolving relatives).

    Returns:
        (inlined_deps, cleaned_source) where inlined_deps is the concatenated
        dependency code to prepend, and cleaned_source is the entry point with
        relative import statements removed.
    """
    source_dir = os.path.dirname(source_path)
    inlined = set()
    return _resolve_local_imports_recursive(source, source_dir, inlined)


def _resolve_local_imports_recursive(source, source_dir, inlined):
    """Recursive implementation of resolve_local_imports."""
    deps_parts = []
    cleaned_lines = []

    for line in source.split("\n"):
        relative_path = _is_relative_import(line)
        if relative_path is not None:
            # Resolve the absolute path of the imported file.
            abs_path = os.path.normpath(os.path.join(source_dir, relative_path))
            if abs_path in inlined:
                # Already inlined by a prior import — just drop the statement.
                continue
            inlined.add(abs_path)

            if not os.path.isfile(abs_path):
                raise FileNotFoundError(
                    "Local import '%s' not found (resolved to %s)"
                    % (relative_path, abs_path)
                )

            with open(abs_path) as f:
                dep_source = f.read()

            # Recursively resolve the dependency's own local imports first.
            dep_dir = os.path.dirname(abs_path)
            dep_inlined, dep_cleaned = _resolve_local_imports_recursive(
                dep_source, dep_dir, inlined
            )
            dep_stripped = _strip_exports(dep_cleaned)
            deps_parts.append(dep_inlined)
            deps_parts.append(
                "// --- Inlined: %s ---\n%s\n" % (relative_path, dep_stripped)
            )
        else:
            cleaned_lines.append(line)

    return "".join(deps_parts), "\n".join(cleaned_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Bundle JS companion modules for a wasm binary."
    )
    parser.add_argument(
        "--wasm",
        required=True,
        help="Path to the .wasm binary (parsed for import section DCE).",
    )
    parser.add_argument(
        "--main",
        required=True,
        help="Path to the entry point JS file.",
    )
    parser.add_argument(
        "--modules",
        required=True,
        help="Path to the modules manifest file (one 'module:path' per line).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output .mjs file.",
    )
    parser.add_argument(
        "--wasm-filename",
        default=None,
        help="Basename of the .wasm binary to inject as __IREE_WASM_BINARY. "
        "If not specified, derived from --wasm path.",
    )
    args = parser.parse_args()

    # Read the modules manifest (JSON array of {module, path} objects).
    with open(args.modules) as f:
        manifest_entries = json.loads(f.read())

    # Parse wasm imports for dead code elimination.
    wasm_import_modules = parse_wasm_import_modules(args.wasm)

    # Group manifest entries by module name, preserving dependency order.
    # The manifest is already in dependency order (from depset traversal).
    seen_modules = {}
    ordered_modules = []
    for entry in manifest_entries:
        module_name = entry["module"]
        path = entry["path"]
        if module_name not in seen_modules:
            seen_modules[module_name] = []
            ordered_modules.append(module_name)
        seen_modules[module_name].append(path)

    # Filter to only modules actually imported by the wasm binary.
    active_modules = [m for m in ordered_modules if m in wasm_import_modules]
    eliminated = [m for m in ordered_modules if m not in wasm_import_modules]
    if eliminated:
        print(
            "DCE: eliminated %d module(s) not imported by wasm binary: %s"
            % (len(eliminated), ", ".join(eliminated)),
            file=sys.stderr,
        )

    # Determine the wasm binary filename for runtime discovery.
    wasm_filename = args.wasm_filename or os.path.basename(args.wasm)

    # Build the output.
    parts = []
    parts.append(
        "// Generated by IREE wasm binary bundler.\n"
        "// DO NOT EDIT — this file is produced by the build system.\n"
        "//\n"
        "// Modules: %s\n"
        "// Wasm binary: %s\n"
        "\n"
        "// Path to the wasm binary, relative to this script.\n"
        "// Entry points use this to locate the .wasm file at runtime.\n"
        "const __IREE_WASM_BINARY = '%s';\n"
        % (", ".join(active_modules), wasm_filename, wasm_filename)
    )

    # Transform and inline each companion module.
    for module_name in active_modules:
        for js_path in seen_modules[module_name]:
            with open(js_path) as f:
                source = f.read()
            parts.append(transform_companion(source, module_name, js_path))

    # Generate the import merger.
    parts.append(generate_merger(active_modules))

    # Resolve and inline the entry point's local imports, then append.
    with open(args.main) as f:
        main_source = f.read()
    main_abs = os.path.abspath(args.main)
    inlined_deps, cleaned_main = resolve_local_imports(main_source, main_abs)
    if inlined_deps:
        parts.append("\n// --- Entry point dependencies ---\n%s" % inlined_deps)
    parts.append("\n// --- Entry point (from %s) ---\n%s" % (args.main, cleaned_main))

    # Write the output.
    with open(args.output, "w") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
