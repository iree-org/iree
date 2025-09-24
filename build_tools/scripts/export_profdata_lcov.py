#!/usr/bin/env python
"""Produces an LCOV file from a directory of LLVM .profraw files.

Finds all .profraw files in a source directory specified as {--base}/ and merges
them into a single .profdata file using llvm-profdata. An optional objects list
specifying absolute paths for .a, .o, or executable binaries will be used to
find the coverage metadata for the collected profiles.

If available a list of CMake targets and their corresponding object file will be
used to try to infer the objects based on the filenames of the .profraw files as
`cmake_target[.optional-extra-discriminators].profraw`. Objects can also be
explicitly specified for any profiles not matching a target.
The `--targets=` file is a ; delimited dictionary of target=path items.
e.g. `cmake_target_a=/bin/a;cmake_target_b=/bin/b`

The paths for llvm-profdata and llvm-cov can be explicitly specified as flags
and otherwise must be available on PATH.

Usage:
  ./build_tools/scripts/export_profdata_lcov.py \\
      --base=../build/coverage/runtime \\
      --targets=../build/coverage/runtime.target.list \\
      --output=../build/coverage/runtime.lcov.info
"""

import argparse
from pathlib import Path
import subprocess
import sys
import os


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        prog="export_profdata_lcov",
        usage=__doc__,
    )
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="Root source directory containing .profraw files.",
    )
    parser.add_argument(
        "--objects",
        type=str,
        help="Optional list of absolute paths for .a, .o, or executable binaries used to find coverage metadata for collected profiles.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        help="Optional semicolon (;) delimited list of [CMake Target]=[path] items, e.g. `cmake_target_a=/bin/a;cmake_target_b=/bin/b`.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output file (e.g. ../build/covarge/runtime.lcov.info).",
    )
    parser.add_argument(
        "--llvm-profdata",
        type=Path,
        default="llvm-profdata",
        help="Path to the llvm-profdata tool (may be relative if on PATH).",
    )
    parser.add_argument(
        "--llvm-cov",
        type=Path,
        default="llvm-cov",
        help="Path to the llvm-cov tool (may be relative if on PATH).",
    )
    args = parser.parse_args(argv)
    return args


def main(args):
    # Erase existing artifacts, if any.
    output_path = args.output
    if output_path.exists():
        print(f"Removing existing {output_path} file")
        output_path.unlink()
    profdata_path = args.base.with_suffix(".profdata")
    if profdata_path.exists():
        print(f"Removing existing {profdata_path} file")
        profdata_path.unlink()

    # Find all .profraw files in the source directory.
    source_dir = Path(f"{args.base}{os.path.sep}")
    if not source_dir.is_dir():
        print(f"WARNING: source directory {source_dir} not found, skipping run")
        sys.exit(0)
    source_files = list(source_dir.glob("*.profraw"))
    if not source_files:
        print(f"No sources profraw files found in {source_dir}, skipping run")
        sys.exit(0)

    # Merge all source profraw files together into a single profdata file.
    print(f"Merging from source files in {source_dir} to {profdata_path}:")
    for source_file in source_files:
        print(f"- {source_file}")
    subprocess.check_call(
        [
            args.llvm_profdata,
            "merge",
            "--sparse",
            f"--output={profdata_path}",
        ]
        + source_files
    )

    # Add any explicitly specified objects first.
    objects = []
    if args.objects:
        objects.append([object for object in args.objects.split(";")])

    # Read target map, if specified.
    # This may not be needed if objects are also specified and is allowed to
    # fail. The file is a ; delimited dictionary of target=path items.
    # e.g. cmake_target_a=/bin/a;cmake_target_b=/bin/b
    target_map = {}
    if args.targets:
        with open(args.targets, "r") as targets_file:
            for target, object in [
                item.split("=", 1) for item in targets_file.read().split(";")
            ]:
                target_map[target] = object

    # Try to automatically add target objects from the source files.
    for source_file in source_files:
        target_name = source_file.stem
        target_object = target_map[target_name]
        if target_object:
            objects.append(target_object)

    if not objects:
        print(f"WARNING: no objects specified/discovered, skipping export")
        print(f"Merged profiling data available in {profdata_path}")
        sys.exit(0)

    # Export the lcov file.
    print(f"Exporting {profdata_path} to {output_path} using objects:")
    for object in objects:
        print(f"- {object}")
    with open(output_path, "wb") as output_file:
        subprocess.check_call(
            [
                args.llvm_cov,
                "export",
                "-format=lcov",
                f"-instr-profile={profdata_path}",
            ]
            + [f"-object={object}" for object in objects],
            stdout=output_file,
            text=False,
        )

    print(f"LCOV file written: {output_path}")

    sys.exit(0)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
