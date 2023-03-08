import argparse

from library import *
from matmul import *
from manifest import *


###############################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates MLIR operations for "\
                     "functional and performance testing of IREE compilers")
  parser.add_argument("--build-dir", default=".", required=True, help="IREE "\
                      "top-level build directory is used to generate operations "\
                      "and npy files")
  parser.add_argument("--operation_kind", default="all", help="Specifies the "\
                      "operation kinds to generate (matmul, conv2d, all)")
  parser.add_argument("--dispatches", default='', help="Comma delimited list to "\
                      "filter dispatches by name. A dispatch is a combination of "\
                      "operation and tuning configuration.")

  parser.add_argument("--mlir-dialect", default='linalg', help='MLIR dialect "\
                      entry point at which operation is emitter. For example, "\
                      "linalg*, mhlo, etc.')


  args = parser.parse_args()
  
  # Manifests metadata for a group of accompanying opeartions and configurations. 
  manifest = Manifest(args)

  # Collect all the pre-defined dispatches in a manifest.
  GpuMatmulTensorCoresF16(manifest)
  GpuMatmulTensorCoresF32(manifest)

  # Emit the dispatches present in MLIR linalg dialect.
  manifest.emit(MlirDialect.Linalg)


