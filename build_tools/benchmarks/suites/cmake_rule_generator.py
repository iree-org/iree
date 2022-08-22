## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator that generates CMake rules from python defined benchmarks.

The rules will build required artifacts to run benchmarks.
"""

from dataclasses import dataclass
from typing import List, Optional
import os
import pathlib
import string
import urllib.parse

from .definitions import common_definitions
from . import iree_benchmarks

TEMPLATE_DIR = pathlib.Path(__file__).parent
DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_download_artifact_template.cmake", "r").read())
TFLITE_IMPORT_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_tflite_import_template.cmake", "r").read())
TF_IMPORT_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_tf_import_template.cmake", "r").read())
IREE_BYTECODE_MODULE_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_bytecode_module_template.cmake", "r").read())
IREE_GENERATE_BENCHMARK_FLAGFILE_CMAKE_TEMPLATE = string.Template(
    open(TEMPLATE_DIR / "iree_generate_benchmark_flagfile_template.cmake",
         "r").read())


@dataclass
class ModelRule(object):
  target_name: str
  file_path: str
  cmake_rule: str


@dataclass
class IreeModelImportRule(object):
  target_name: str
  output_file_path: str
  mlir_dialect_type: str
  cmake_rule: Optional[str]


@dataclass
class IreeModuleCompileRule(object):
  target_name: str
  output_module_path: str
  cmake_rule: str


@dataclass
class IreeGenerateBenchmarkFlagfileRule(object):
  target_name: str
  output_flagfile_path: str
  cmake_rule: str


class CommonRuleFactory(object):
  """Generates common cmake rules."""

  def __init__(self, model_artifacts_dir: str):
    """Constructs a CommonRuleFactory.

    Args:
      model_artifacts_dir: root directory to store model files. Can contain
      CMake variable syntax in the path.
    """
    self._model_artifacts_dir = model_artifacts_dir
    self._model_rules = {}

  def add_model_rule(self, model: common_definitions.Model) -> ModelRule:
    """Adds a rule to fetch a model. Reuses the existing rule when possible."""
    if model.id in self._model_rules:
      return self._model_rules[model.id]

    # Model target: <package_name>-model-<model_id>
    target_name = f"model-{model.id}"

    model_url = urllib.parse.urlparse(model.source_url)
    _, file_ext = os.path.splitext(model_url.path)
    # Model path: <model_artifacts_dir>/<model_id>_<model_name>.<file ext>
    model_path = f"{self._model_artifacts_dir}/{model.id}_{model.name}{file_ext}"

    if model_url.scheme == "https":
      cmake_rule = DOWNLOAD_ARTIFACT_CMAKE_TEMPLATE.substitute(
          __TARGET_NAME=target_name,
          __OUTPUT_PATH=model_path,
          __SOURCE_URL=model.source_url)
    else:
      raise ValueError("Unsupported model url: {model.source_url}.")

    model_rule = ModelRule(target_name=target_name,
                           file_path=model_path,
                           cmake_rule=cmake_rule)

    self._model_rules[model.id] = model_rule
    return model_rule

  def generate_cmake_rules(self) -> List[str]:
    """Dump all cmake rules in a correct order."""
    return [rule.cmake_rule for rule in self._model_rules.values()]


class IreeRuleFactory(object):
  """Generates IREE benchmark cmake rules."""

  def __init__(self, iree_artifacts_dir):
    """Constructs an IreeRuleFactory.

    Args:
      iree_artifacts_dir: root directory to store generated IREE artifacts. Can
        contain CMake variable syntax in the path.
    """
    self._iree_artifacts_dir = iree_artifacts_dir
    self._import_model_rules = {}
    self._compile_module_rules = {}
    self._generate_flagfile_rules = {}

  def add_import_model_rule(
      self,
      model_id: str,
      model_name: str,
      model_source_type: common_definitions.ModelSourceType,
      source_model_rule: ModelRule,
  ) -> IreeModelImportRule:
    """Adds a rule to fetch the model and import into MLIR. Reuses the rule when
    possible."""

    if model_id in self._import_model_rules:
      return self._import_model_rules[model_id]

    # If the source model is MLIR, no import rule is needed.
    if model_source_type == common_definitions.ModelSourceType.EXPORTED_LINALG_MLIR:
      import_model_rule = IreeModelImportRule(
          target_name=source_model_rule.target_name,
          output_file_path=source_model_rule.file_path,
          mlir_dialect_type="linalg",
          cmake_rule=None)
      self._import_model_rules[model_id] = import_model_rule
      return import_model_rule

    # Import target: <package_name>_iree-import-model-<model_id>
    target_name = f"iree-import-model-{model_id}"

    # Imported MLIR path: <iree_artifacts_dir>/<model_id>_<model_name>/<model_name>.mlir
    output_file_path = f"{self._iree_artifacts_dir}/{model_id}_{model_name}/{model_name}.mlir"

    if model_source_type == common_definitions.ModelSourceType.EXPORTED_TFLITE:
      cmake_rule = TFLITE_IMPORT_CMAKE_TEMPLATE.substitute(
          __TARGET_NAME=target_name,
          __SOURCE_MODEL_PATH=source_model_rule.file_path,
          __OUTPUT_PATH=output_file_path)
      mlir_dialect_type = "tosa"
    elif model_source_type == common_definitions.ModelSourceType.EXPORTED_TF:
      cmake_rule = TF_IMPORT_CMAKE_TEMPLATE.substitute(
          __TARGET_NAME=target_name,
          __SOURCE_MODEL_PATH=source_model_rule.file_path,
          __OUTPUT_PATH=output_file_path)
      mlir_dialect_type = "mhlo"
    else:
      raise ValueError(
          f"Unsupported source type '{model_source_type}' of the model '{model_id}'."
      )

    import_model_rule = IreeModelImportRule(target_name=target_name,
                                            output_file_path=output_file_path,
                                            mlir_dialect_type=mlir_dialect_type,
                                            cmake_rule=cmake_rule)

    self._import_model_rules[model_id] = import_model_rule
    return import_model_rule

  def add_compile_module_rule(
      self, compile_spec: iree_definitions.BenchmarkCompileSpec,
      model_import_rule: IreeModelImportRule, compile_flags: List[str]):
    """Adds a rule to compile a MLIR into a IREE module. Reuses the existing
    rule when possible."""

    model = compile_spec.model
    compile_config = compile_spec.compile_config

    target_id = f"{model.id}-{compile_config.id}"
    if target_id in self._compile_module_rules:
      return self._compile_module_rules[target_id]

    # Module target: <package_name>_iree-module-<model_id>-<compile_config_id>
    target_name = f"iree-module-{target_id}"

    # Module path: <iree_artifacts_dir>/<model_id>_<model_name>/<compile_config_id>.vmfb
    output_path = f"{self._iree_artifacts_dir}/{model.id}_{model.name}/{compile_config.id}.vmfb"

    cmake_rule = IREE_BYTECODE_MODULE_CMAKE_TEMPLATE.substitute(
        __TARGET_NAME=target_name,
        __MODULE_OUTPUT_PATH=output_path,
        __SOURCE_MODEL_PATH=model_import_rule.output_file_path,
        __COMPILE_FLAGS=";".join(compile_flags),
        __SOURCE_MODEL_TARGET=model_import_rule.target_name)
    compile_module_rule = IreeModuleCompileRule(target_name=target_name,
                                                output_module_path=output_path,
                                                cmake_rule=cmake_rule)

    # TODO(10155): Dump the compile flags from iree_bytecode_module into a flagfile.

    self._compile_module_rules[target_id] = compile_module_rule
    return compile_module_rule

  def add_generate_benchmark_flagfile_rule(
      self, run_spec: iree_definitions.BenchmarkRunSpec,
      compile_module_rule: IreeModuleCompileRule
  ) -> IreeGenerateBenchmarkFlagfileRule:
    """Adds a rule to dump the flags to run the benchmark. Reuses the existing
    rule when possible."""

    compile_spec = run_spec.compile_spec
    run_config = run_spec.run_config
    compile_config = compile_spec.compile_config
    model = compile_spec.model

    if run_config.benchmark_tool != iree_benchmarks.MODULE_BENCHMARK_TOOL:
      raise ValueError(
          f"Benchmark tool '{run_config.benchmark_tool}' is not supported yet.")

    target_id = f"{model.id}-{compile_config.id}-{run_config.id}"
    if target_id in self._generate_flagfile_rules:
      return self._generate_flagfile_rules[target_id]

    # Flagfile target: <package_name>_iree-benchmark-flagfile-<model_id>-<compile_config_id>-<run_config_id>
    target_name = f"iree-benchmark-flagfile-{target_id}"

    output_dir = f"{self._iree_artifacts_dir}/{model.id}_{model.name}"
    # Flagfile path: <iree_artifacts_dir>/<model_id>_<model_name>/<compile_config_id>_<run_config_id>.flagfile
    output_path = f"{output_dir}/{compile_config.id}_{run_config.id}.flagfile"

    cmake_rule = IREE_GENERATE_BENCHMARK_FLAGFILE_CMAKE_TEMPLATE.substitute(
        __TARGET_NAME=target_name,
        __OUTPUT_PATH=output_path,
        __MODULE_FILE_PATH=compile_module_rule.output_module_path,
        # Use relative path in the flagfile to make it portable.
        __REL_MODULE_FILE_PATH=os.path.relpath(
            compile_module_rule.output_module_path, output_dir),
        __DRIVER=run_config.driver.value,
        __ENTRY_FUNCTION=model.entry_function,
        __FUNCTION_INPUTS=",".join(model.input_types),
        __EXTRA_FLAGS=";".join(run_config.extra_flags))
    generate_flagfile_rule = IreeGenerateBenchmarkFlagfileRule(
        target_name=target_name,
        output_flagfile_path=output_path,
        cmake_rule=cmake_rule)

    self._generate_flagfile_rules[target_id] = generate_flagfile_rule
    return generate_flagfile_rule

  def generate_cmake_rules(self) -> List[str]:
    """Dump all cmake rules in a correct order."""
    import_model_rules = [
        rule.cmake_rule for rule in self._import_model_rules.values()
    ]
    compile_module_rules = [
        rule.cmake_rule for rule in self._compile_module_rules.values()
    ]
    generate_flagfile_rules = [
        rule.cmake_rule for rule in self._generate_flagfile_rules.values()
    ]
    return import_model_rules + compile_module_rules + generate_flagfile_rules


def _generate_iree_compile_target_flags(
    target: iree_definitions.CompileTarget) -> List[str]:
  arch_info: common_definitions.ArchitectureInfo = target.target_architecture.value
  if arch_info.architecture == "x86_64":
    flags = [
        f"--iree-llvm-target-triple=x86_64-unknown-{target.target_platform.value}",
        f"--iree-llvm-target-cpu={arch_info.microarchitecture.lower()}"
    ]
  else:
    raise ValueError("Unsupported architecture.")
  return flags


def _generate_iree_compile_flags(compile_config: iree_definitions.CompileConfig,
                                 mlir_dialect_type: str) -> List[str]:
  if len(compile_config.compile_targets) != 1:
    raise ValueError("Only support one compile target for now.")

  compile_target = compile_config.compile_targets[0]
  flags = [
      f"--iree-hal-target-backends={compile_target.target_backend.value}",
      f"--iree-input-type={mlir_dialect_type}"
  ]
  flags.extend(_generate_iree_compile_target_flags(compile_target))
  return flags


def _generate_iree_benchmark_rules(common_rule_factory: CommonRuleFactory,
                                   iree_artifacts_dir: str) -> List[str]:
  iree_rule_factory = IreeRuleFactory(iree_artifacts_dir)
  compile_specs, run_specs = iree_benchmarks.Linux_x86_64_Benchmarks.generate()
  compile_spec_map = {}
  for compile_spec in compile_specs:
    model = compile_spec.model
    compile_config = compile_spec.compile_config

    source_model_rule = common_rule_factory.add_model_rule(model)
    import_rule = iree_rule_factory.add_import_model_rule(
        model_id=model.id,
        model_name=model.name,
        model_source_type=model.source_type,
        source_model_rule=source_model_rule)

    compile_flags = _generate_iree_compile_flags(
        compile_config=compile_spec.compile_config,
        mlir_dialect_type=import_rule.mlir_dialect_type)
    compile_rule = iree_rule_factory.add_compile_module_rule(
        compile_spec=compile_spec,
        model_import_rule=import_rule,
        compile_flags=compile_flags)
    compile_spec_map[(model.id, compile_config.id)] = compile_rule

  for run_spec in run_specs:
    compile_spec = run_spec.compile_spec
    compile_rule = compile_spec_map[(compile_spec.model.id,
                                     compile_spec.compile_config.id)]
    iree_rule_factory.add_generate_benchmark_flagfile_rule(
        run_spec, compile_rule)

  return iree_rule_factory.generate_cmake_rules()


def generate_benchmark_rules(model_artifacts_dir: str,
                             iree_artifacts_dir: str) -> List[str]:
  """Generates cmake rules for all benchmarks.
  
  Args:
    model_artifacts_dir: root directory to store model files. Can contain CMake
      variable syntax in the path.
    iree_artifacts_dir: root directory to store generated IREE artifacts. Can
      contain CMake variable syntax in the path.
  Returns:
    List of CMake rules.
  """
  common_rule_factory = CommonRuleFactory(model_artifacts_dir)
  iree_rules = _generate_iree_benchmark_rules(common_rule_factory,
                                              iree_artifacts_dir)
  # Currently the rules are simple so the common rules can be always put at the
  # top. Need a topological sort once the dependency gets complicated.
  return common_rule_factory.generate_cmake_rules() + iree_rules
