import argparse

from library import *
from matmul import *
from manifest import *

###############################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates MLIR operations for "\
                     "verification and profiling of IREE compiled dispatches.")
  parser.add_argument("--build-dir", default=".", \
                      help="IREE top-level build directory is used to generate "\
                        "operations and npy files")
  parser.add_argument("--operation_kind", default="all", \
                      help="Specifies the operation kinds to generate.", \
                      choices=["matmul", "conv2d", "all"])
  parser.add_argument("--dispatches", default='', help="Comma delimited list to "\
                      "filter dispatches by name. A dispatch is a combination of "\
                      "operation and tuning configuration.")
  parser.add_argument("--mlir-dialect",\
                      default='linalg',\
                      help="MLIR dialect entry point at which operation is emitter.", \
                      choices=["linalg", "flow", "all"])

  args = parser.parse_args()

  # Manifests dispatches for a group of accompanying operations and configurations.
  manifest = Manifest(args)

  # Collect all the pre-defined dispatches in a manifest for various operations.
  gpu_matmul_tensor_cores_f16(manifest)
  #gpu_matmul_tensor_cores_f32(manifest)

  # Emit the dispatches in MLIR source files.
  manifest.emit(MlirDialect.Linalg)
