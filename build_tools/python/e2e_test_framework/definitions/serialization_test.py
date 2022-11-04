import collections
import json
from e2e_test_framework.definitions import common_definitions, iree_definitions, serialization
from e2e_test_framework.models import tflite_models


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
  config = iree_definitions.ModuleExecutionConfig(
      id="abcd",
      tags=[],
      loader=iree_definitions.RuntimeLoader.EMBEDDED_ELF,
      driver=iree_definitions.RuntimeDriver.LOCAL_SYNC,
      tool="test",
      extra_flags=[])

  container_map = collections.OrderedDict()
  print(
      json.dumps(gen_config,
                 cls=serialization.CustomEncoder,
                 container_map=container_map))
  print(container_map)


if __name__ == "__main__":
  main()
