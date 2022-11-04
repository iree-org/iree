import collections
import dataclasses
import json
import typing
from e2e_test_framework import serialization
from e2e_test_framework.definitions import common_definitions, iree_definitions
from e2e_test_framework.models import tflite_models
from benchmark_suites.iree import benchmark_collections


def main():
  compile_config = iree_definitions.CompileConfig(
      id="1234",
      tags=["a", "b"],
      compile_targets=[
          iree_definitions.CompileTarget(
              target_backend=iree_definitions.TargetBackend.LLVM_CPU,
              target_architecture=common_definitions.DeviceArchitecture.
              RV64_GENERIC,
              target_abi=iree_definitions.TargetABI.LINUX_GNU)
      ])
  gen_config = iree_definitions.ModuleGenerationConfig(
      imported_model=iree_definitions.ImportedModel.from_model(
          tflite_models.MOBILENET_V2),
      compile_config=compile_config)
  exec_config = iree_definitions.ModuleExecutionConfig(
      id="abcd",
      tags=[],
      loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
      driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
      tool="test",
      extra_flags=[])

  _, run_configs = benchmark_collections.generate_benchmarks()

  serialized_obj_map = collections.OrderedDict()
  serialized_obj = serialization.serialize(run_configs, serialized_obj_map)
  objs = serialization.deserialize(
      serialized_obj, typing.List[iree_definitions.E2EModelRunConfig],
      serialized_obj_map)
  for a, b in zip(run_configs, objs):
    print(a)
    print(b)
  print(run_configs == objs)


if __name__ == "__main__":
  main()
