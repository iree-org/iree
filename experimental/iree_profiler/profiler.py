import argparse

from library import *
from matmul import *
from manifest import *
from performance_report import *

operation_launcher_map = {
  OperationKind.Matmul : MatmulOperationLauncher,
}

if __name__ == "__main__":
  ###############################################################################
  # Parse command line arguments
  ###############################################################################
  parser = argparse.ArgumentParser(description="IREE Python profiler tool for "\
                                   "verifcation and performance profiling tool "\
                                    "for IREE-compiled MLIR operations.")
  
  # General options 
  parser.add_argument("--build-dir", default=".", required=True, \
                      help="IREE top-level build directory is used to generate "\
                        "operations and npy files.This should be same that used "\
                        "to call generated.py")
  parser.add_argument("--verbose", default='False', \
                      help='Prints verbose output and commands executed.')
  
  # Generator options
  parser.add_argument("--operation_kind", default="all", help="Specifies the "\
                      "operation kinds to generate (matmul, conv2d, all)")
  parser.add_argument("--dispatches", default='', help="Comma delimited list to "\
                      "filter dispatches by name. A dispatch is a combination of "\
                      "operation and tuning configuration.")
  parser.add_argument("--mlir-dialect", default='linalg', help='MLIR dialect entry "\
                      "point at which operation is emitter. For example, "\
                      "linalg*, mhlo, etc.')
  # Compilation options
  parser.add_argument("--device", default="cuda", \
                      help="Target backend device to benchmark the operation on. "\
                        "For example, cuda, vulkan, etc.")
  parser.add_argument("--force-compile", default='False', \
                      type=str, help="Force re-compilation of the operation even "\
                      "if .vmfb file is present.")
  parser.add_argument("--compile-only", default='False', \
                      type=str, help="Compiles the operation "\
                        "without running verification and profiling.")

  # Profiling options
  parser.add_argument("--profiling-enabled", "--benchmark", default='True', \
                      type=str, help="Benchmark the operation.")
  parser.add_argument('--batch-size', '--benchmark-dispatch-repeat-count', \
                      default=100, help="Number of times dispatch is launched "\
                        "in a loop to amortize the launch overhead.")
  parser.add_argument("--benchmark-repetitions", default=5, 
                      type=int, help="Number of times benchmark is repeated "\
                      "and min, max, median, and average runtimes/gflops are "\
                      "reported.")

  # Verification options 
  parser.add_argument("--verification-enabled", default='True', 
                      type=str, help="Verify the operation against reference numpy "\
                      "implementation.")
  
  # Performance reporting options
  parser.add_argument("--output", default='', \
                      help="Path to output file for csv readable results.")
  parser.add_argument("--append", default='false', \
                      help="If true, result is appended to possibly existing file. "\
                        "Otherwise, any existing file is overwritten.")
  
  parser.add_argument("--tags", default='', \
                      help="Inserts leading columns in output table and uniform "\
                        "values for each column. Useful for generating pivot tables.") 
  args = parser.parse_args()
  ###############################################################################

  # Boolenize the string arguments from command line.
  verification_enabled = False if args.verification_enabled in ['False', 'false', '0'] else True
  profiling_enabled = False if args.profiling_enabled in ['False', 'false', '0'] else True
  compile_only = False if args.compile_only in ['False', 'false', '0'] else True
  # Overrite verification and profiling if compile_only is set.
  if compile_only:
    verification_enabled = False
    profiling_enabled = False

  # Path to the directory where the generated operations are stored.
  generated_path = os.path.join(args.build_dir, 'generated', args.mlir_dialect)

  # Manifests metadata for a group of accompanying opeartions and configurations. 
  manifest = Manifest(args)

  # Collect all the avialable operations in a manifest.
  GpuMatmulTensorCoresF16(manifest)
  #GpuMatmulTensorCoresF32(manifest)

  # Performance report
  perf_report = PerformanceReport(args)

  # For all the operations in the manifest, compile and profile them.
  for operation_kind, operation_collection_list in manifest.operations.items():
    for operation_collection in operation_collection_list:
      
      # Select and create an instance of operation_launcher for the operation with operation_kind.
      # print(operation_collection.operation.name())
      operation_launcher = operation_launcher_map[operation_kind](args, operation_collection.operation)

      for configuration in operation_collection.configuration_list:
        if verification_enabled:
          operation_launcher.verify(configuration)
        if profiling_enabled:
          runtime = operation_launcher.profile(configuration)
          
          # Create and print a performance result.
          result = PerformanceResult(operation_collection.operation, configuration, runtime)
          result.print()

          # Append the performance result to the performance report.
          perf_report.append_perf_result(result)

      # Compile the operation for verification and profiling.
      if compile_only:
        operation_launcher.compile(CompilationMode.Verify)
        operation_launcher.compile(CompilationMode.Benchmark)

  # Write the performance report to a csv file.
  if args.output != '':
    perf_report.write_csv()