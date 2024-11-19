# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.build.executor import BuildAction, BuildContext, BuildFile, BuildFileLike
from iree.build.metadata import CompileSourceMeta

__all__ = [
    "onnx_import",
]


def onnx_import(
    *,
    # Name of the rule and output of the final artifact.
    name: str,
    # Source onnx file.
    source: BuildFileLike,
    upgrade: bool = True,
) -> BuildFile:
    context = BuildContext.current()
    input_file = context.file(source)
    output_file = context.allocate_file(name)

    ImportOnnxAction(
        input_file=input_file,
        output_file=output_file,
        upgrade=upgrade,
        desc=f"Importing ONNX {name} -> {output_file}",
        executor=context.executor,
        deps=[
            input_file,
        ],
    )

    return output_file


class ImportOnnxAction(BuildAction):
    def __init__(
        self, input_file: BuildFile, output_file: BuildFile, upgrade: bool, **kwargs
    ):
        super().__init__(**kwargs)
        self.input_file = input_file
        self.output_file = output_file
        self.upgrade = upgrade
        self.deps.add(input_file)
        output_file.deps.add(self)
        CompileSourceMeta.get(output_file).input_type = "onnx"

    def _invoke(self):
        import iree.compiler.tools.import_onnx.__main__ as m

        args = [
            str(self.input_file.get_fs_path()),
            "-o",
            str(self.output_file.get_fs_path()),
        ]
        if self.upgrade:
            args.extend(["--opset-version", "17"])
        parsed_args = m.parse_arguments(args)
        m.main(parsed_args)
