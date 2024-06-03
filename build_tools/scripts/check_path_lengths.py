#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This scans the IREE source tree for long path lengths, which are problematic
# on Windows: https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
#
# We ultimately care that the build system is happy, but CMake on Windows in
# particular does not actually give early or easy to understand error messages,
# and developers/CI using Linux may still want to see warnings. We'll use
# relative directory path length as a reasonable heuristic for "will the build
# system be happy?", since CMake tends to create paths like this:
# `iree/compiler/.../Foo/CMakeFiles/iree_compiler_Foo_Foo.objects.dir/bar.obj`.
# Note that 'Foo' appears three times in that path, so that's typically the best
# place to trim characters (and not file names).
#
# To check that all relative paths are shorter than the default limit:
#   python check_path_lengths.py
#
# To check that all relative paths are shorter than a custom limit:
#   python check_path_lengths.py --limit=50

import argparse
import os
import pathlib
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description="Path length checker")
    # The default limit was selected based on repository state when this script
    # was added. If the max path length decreases, consider lowering this too.
    parser.add_argument(
        "--limit", help="Path length limit (inclusive)", type=int, default=75
    )
    parser.add_argument(
        "--include_tests",
        help="Includes /test directories. False by default as these don't usually generate problematic files during the build",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Outputs detailed information about path lengths",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


def main(args):
    repo_root = pathlib.Path(__file__).parent.parent.parent

    # Just look at the compiler directory for now, since it has historically had
    # by far the longest paths.
    walk_root = os.path.join(repo_root, "compiler")

    longest_path_length = -1
    long_paths = []
    short_paths = []
    for dirpath, dirnames, _ in os.walk(walk_root):
        # Don't descend into test directories, since they typically don't generate
        # object files or binaries that could trip up the build system.
        if not args.include_tests and "test" in dirnames:
            dirnames.remove("test")
        # Skip build directories (should really anything be covered by .gitignore).
        if "build" in dirnames:
            dirnames.remove("build")

        path = pathlib.Path(dirpath).relative_to(repo_root).as_posix()
        if len(path) > args.limit:
            long_paths.append(path)
        else:
            short_paths.append(path)
        longest_path_length = max(longest_path_length, len(path))
    long_paths.sort(key=len)
    short_paths.sort(key=len)

    if args.verbose and short_paths:
        print(f"These paths are shorter than the limit of {args.limit} characters:")
        for path in short_paths:
            print("{:3d}, {}".format(len(path), path))

    if long_paths:
        print(f"These paths are longer than the limit of {args.limit} characters:")
        for path in long_paths:
            print("{:3d}, {}".format(len(path), path))
        print(
            f"Error: {len(long_paths)} source paths are longer than {args.limit} characters."
        )
        print("  Long paths can be problematic when building on Windows.")
        print("  Please look at the output above and trim the paths.")
        sys.exit(1)
    else:
        print(f"All path lengths are under the limit of {args.limit} characters.")


if __name__ == "__main__":
    main(parse_arguments())
