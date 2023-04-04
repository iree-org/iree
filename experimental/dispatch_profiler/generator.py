import argparse

from library import *
from matmul import *
from manifest import *
from options import parse_generator_arguments

###############################################################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Generates MLIR operations for "\
                     "verification and profiling of IREE compiled dispatches.")

  args = parse_generator_arguments(parser)

  # Manifests dispatches for a group of accompanying operations and configurations.
  manifest = Manifest(args)

  # Load all the pre-defined dispatches in a manifest.
  manifest.load()

  # Emit the dispatches in MLIR source files.
  manifest.emit(MlirDialect.Linalg)
