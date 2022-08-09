## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from string import Template
from typing import List
from urllib.parse import urlparse
import os

from benchmarks.definitions.common import ArchitectureInfo, ModelSourceType
from benchmarks import iree_benchmarks
from benchmarks.definitions.iree import IreeCompileConfig, IreeCompileTarget

TEMPLATE_DIR = Path(__file__).parent
DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE = Template(
    open(TEMPLATE_DIR / "download_artifact_template.cmake", "r").read())
TFLITE_IMPORT_CMAKE_TEMPLATE = Template(
    open(TEMPLATE_DIR / "tflite_import_template.cmake", "r").read())
IREE_BYTECODE_MODULE_CMAKE_TEMPLATE = Template(
    open(TEMPLATE_DIR / "iree_bytecode_module_template.cmake", "r").read())


def generate_compile_target_flags(target: IreeCompileTarget) -> List[str]:
  arch_info: ArchitectureInfo = target.target_architecture.value
  if arch_info.architecture == "x86_64":
    flags = [
        f"--iree-llvm-target-triple=x86_64-unknown-{target.target_platform.value}",
        f"--iree-llvm-target-cpu={arch_info.microarchitecture.lower()}"
    ]
  else:
    raise ValueError("Unsupported architecture.")
  return flags


def generate_compile_flags(compile_config: IreeCompileConfig) -> List[str]:
  if len(compile_config.compile_targets) != 1:
    raise ValueError("Only support one compile target for now.")
  compile_target = compile_config.compile_targets[0]
  flags = [
      "--iree-mlir-to-vm-bytecode-module",
      f"--iree-hal-target-backends={compile_target.target_backend.value}"
  ]
  flags.extend(generate_compile_target_flags(compile_target))
  return flags


def main():
  compile_specs, runs = iree_benchmarks.generate_iree_benchmarks()
  models = [compile_spec.model for compile_spec in compile_specs]
  generated_model_targets = {}
  for model in models:
    if model.id in generated_model_targets:
      continue

    model_url = urlparse(model.source_uri)
    if model_url.scheme == "https":
      # Local path: root_dir/<model_id>_<artifact basename>
      artifact_local_path = f"${{_ROOT_ARTIFACTS_DIR}}/{model.id}_{os.path.basename(model_url.path)}"
      download_model_rule = DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE.substitute(
          _ARTIFACT_LOCAL_PATH_=artifact_local_path,
          _ARTIFACT_SOURCE_URL_=model.source_uri)
      print(download_model_rule)
    else:
      raise ValueError("Unsupported model url: {model.source_uri}.")

    if model.source_type == ModelSourceType.EXPORTED_MLIR:
      mlir_local_path = artifact_local_path
      model_cmake_target = mlir_local_path
    elif model.source_type == ModelSourceType.EXPORTED_TFLITE:
      # Import target: <package_name>_iree-import-tf-<model_name>-<model_id>
      model_cmake_target = f"${{_PACKAGE_NAME}}_iree-import-tf-{model.name}-{model.id}"
      # Imported MLIR local path: root_dir/<model_id>_<model_name>.mlir
      mlir_local_path = f"${{_ROOT_ARTIFACTS_DIR}}/{model.id}_{model.name}.mlir"
      import_rule = TFLITE_IMPORT_CMAKE_TEMPLATE.substitute(
          _IMPORT_TARGET_=model_cmake_target,
          _ARTIFACT_LOCAL_PATH_=artifact_local_path,
          _OUTPUT_PATH_=mlir_local_path)
      print(import_rule)
    else:
      raise ValueError(f"Unsupported model source: {model.source_type}.")

    generated_model_targets[model.id] = (model_cmake_target, mlir_local_path)

  for compile_spec in compile_specs:
    compile_flags = generate_compile_flags(compile_spec.compile_config)
    model_id = compile_spec.model.id
    config_id = compile_spec.compile_config.id
    model_cmake_target, mlir_local_path = generated_model_targets[model_id]
    module_rule = IREE_BYTECODE_MODULE_CMAKE_TEMPLATE.substitute(
        _MODULE_TARGET_NAME_=f"model-{model_id}-compile-{config_id}",
        _OUTPUT_PATH_=
        f"${{_VMFB_ARTIFACTS_DIR}}/model_{model_id}_compile_{config_id}.vmfb",
        _MLIR_LOCAL_PATH_=mlir_local_path,
        _COMPILE_FLAGS_=";".join(f"\"{flag}\"" for flag in compile_flags),
        _SOURCE_MODEL_TARGET_=model_cmake_target)
    print(module_rule)


if __name__ == "__main__":
  main()
