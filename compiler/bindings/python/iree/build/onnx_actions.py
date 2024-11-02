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

    # Chain through an upgrade if requested.
    if upgrade:
        processed_file = context.allocate_file(f"{name}__upgrade.onnx")
        UpgradeOnnxAction(
            input_file=input_file,
            output_file=processed_file,
            executor=context.executor,
            desc=f"Upgrading ONNX {input_file} -> {processed_file}",
            deps=[
                input_file,
            ],
        )
        input_file = processed_file

    # Import.
    ImportOnnxAction(
        input_file=input_file,
        output_file=output_file,
        desc=f"Importing ONNX {name} -> {output_file}",
        executor=context.executor,
        deps=[
            input_file,
        ],
    )

    return output_file


class UpgradeOnnxAction(BuildAction):
    def __init__(self, input_file: BuildFile, output_file: BuildFile, **kwargs):
        super().__init__(**kwargs)
        self.input_file = input_file
        self.output_file = output_file
        self.deps.add(self.input_file)
        output_file.deps.add(self)
        CompileSourceMeta.get(output_file).input_type = "onnx"

    def _invoke(self):
        import onnx

        input_path = self.input_file.get_fs_path()
        output_path = self.output_file.get_fs_path()

        original_model = onnx.load_model(str(input_path))
        converted_model = onnx.version_converter.convert_version(original_model, 17)
        onnx.save(converted_model, str(output_path))


class ImportOnnxAction(BuildAction):
    def __init__(self, input_file: BuildFile, output_file: BuildFile, **kwargs):
        super().__init__(**kwargs)
        self.input_file = input_file
        self.output_file = output_file
        self.deps.add(input_file)
        output_file.deps.add(self)
        CompileSourceMeta.get(output_file).input_type = "onnx"

    def _invoke(self):
        import iree.compiler.tools.import_onnx.__main__ as m

        args = m.parse_arguments(
            [
                str(self.input_file.get_fs_path()),
                "-o",
                str(self.output_file.get_fs_path()),
            ]
        )
        m.main(args)
