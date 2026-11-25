# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Console tool for importing saved torch programs (``.pt2``) into IREE.

Typical usage:

  iree-import-torch model.pt2 -o model.mlir
  iree-import-torch prefill=prefill.pt2 decode=decode.pt2 -o model.mlir \\
      --externalize-params --save-params-to model.irpa

Each input becomes one function of the module. An input without a name
becomes the function ``main``.
"""

import argparse
import importlib
import sys
from pathlib import Path

from iree.importers.torch.pt2 import import_programs, load_pt2


def _parse_input(text: str) -> tuple[str, Path]:
    name, separator, path = text.partition("=")
    if not separator:
        return "main", Path(text)
    if not name.isidentifier():
        raise argparse.ArgumentTypeError(f"{name!r} is not a valid function name")
    return name, Path(path)


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iree-import-torch",
        description="Imports saved torch programs (.pt2) into IREE.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=_parse_input,
        metavar="[NAME=]PT2",
        help="Program to import, optionally with the name of its function",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        required=True,
        help="Output file, written as bytecode if it ends in .mlirbc",
    )
    parser.add_argument(
        "--output-type",
        choices=["iree", "torch"],
        default="iree",
        help="Whether to write IREE input IR (default) or the torch dialect",
    )
    parser.add_argument(
        "--sync-entry-points",
        action="store_true",
        help="Do not emit asynchronous entry points (only for --output-type=iree)",
    )
    parser.add_argument(
        "--module-name",
        default=None,
        help="Name of the module",
    )
    parser.add_argument(
        "--externalize-params",
        action="store_true",
        help="Reference parameters by name instead of embedding their contents",
    )
    parser.add_argument(
        "--save-params-to",
        type=Path,
        default=None,
        help="Parameter archive (.irpa) to write the externalized parameters to",
    )
    parser.add_argument(
        "--no-decompose",
        action="store_true",
        help="Import the programs as saved, without the import decompositions",
    )
    parser.add_argument(
        "--no-share-tensors",
        action="store_true",
        help="Do not share the tensors that the programs have in common",
    )
    parser.add_argument(
        "--import-module",
        action="append",
        default=[],
        metavar="MODULE",
        help="Python module to import before loading, for example one that "
        "registers the custom operators or pytree types the programs use. "
        "May be repeated.",
    )
    return parser


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line of the tool, reporting errors as argparse does."""
    parser = _create_argument_parser()
    args = parser.parse_args(argv)

    names = [name for name, _ in args.inputs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        parser.error(f"function names used more than once: {', '.join(duplicates)}")
    if args.save_params_to is not None and not args.externalize_params:
        parser.error("--save-params-to requires --externalize-params")
    return args


def main(args: argparse.Namespace) -> int:
    """Import the programs as ``parse_arguments`` describes them."""
    for module in args.import_module:
        importlib.import_module(module)

    programs = {name: load_pt2(path) for name, path in args.inputs}
    output = import_programs(
        programs,
        module_name=args.module_name,
        decompose=not args.no_decompose,
        share_tensors=not args.no_share_tensors,
        externalize_parameters=args.externalize_params,
        save_parameters_to=args.save_params_to,
    )
    if args.output_type == "iree":
        output.lower_to_iree(emit_async_entry_points=not args.sync_entry_points)
    output.save_mlir(args.output_file)
    return 0


def _cli_main() -> None:
    sys.exit(main(parse_arguments()))


if __name__ == "__main__":
    _cli_main()
