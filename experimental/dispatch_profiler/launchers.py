from library import *
from batch_matmul import *
import os.path
import subprocess


class IreeToolsLauncher:
  """Launcher for IREE tools."""

  def __init__(self, args, operation):
    self.operation = operation

    self.generated_path = os.path.join(args.build_dir, 'generated',
                                       args.mlir_dialect)
    self.args = args
    self.benchmark_dispatch_repeat_count = args.batch_size
    self.batch_size = args.batch_size

    # paths to source dispatch mlir, compiled vmfb, and logs.
    self.operation_path = os.path.join(
        self.generated_path, OperationKindNames[operation.operation_kind],
        operation.name())
    self.source_mlir_file = os.path.join(self.operation_path,
                                         operation.name() + '.mlir')

    # path to cached numpy refernece input/output files.
    self.op_reference_cache_path = os.path.join(args.build_dir, 'generated',\
                                             'reference_cache', operation.name())

    if not os.path.exists(self.op_reference_cache_path):
      os.makedirs(self.op_reference_cache_path)

    # path to iree-compile tool. (for compiling the input mlir file to vmfb)
    self.iree_compile_path = os.path.join(args.build_dir, 'tools',
                                          'iree-compile')

    # path to iree-benchmark-module tool. (for performance benchmarking and profiling)
    self.iree_benchmark_module_path = os.path.join(args.build_dir, 'tools',
                                                   'iree-benchmark-module')

    # path to iree-run-module tool. (for verification)
    self.iree_run_module_path = os.path.join(args.build_dir, 'tools',
                                             'iree-run-module')

    # output vmfb files for the operation.
    self.vmfb_verify_file = os.path.join(self.operation_path,
                                         self.operation.name() + '_verify.vmfb')
    self.vmfb_benchmark_file = os.path.join(
        self.operation_path,
        self.operation.name() + '_benchmark.vmfb')

    # reference implementation for the operation_kind.
    self.reference_impl_map = {
        OperationKind.Matmul: ReferenceMatmulOp,
        OperationKind.BatchMatmul: ReferenceBatchMatmulOp,
    }

  def iree_compile(self, compilation_mode):
    """Compiles the input mlir file to vmfb file."""

    benchmark_dispatch_repeat_count = self.benchmark_dispatch_repeat_count if compilation_mode == CompilationMode.Profile else 1
    vmfb_file = self.vmfb_benchmark_file if compilation_mode == CompilationMode.Profile else self.vmfb_verify_file

    # Base iree-compile commandline
    cmd = [self.iree_compile_path, self.source_mlir_file, "-o", f"{vmfb_file}"]

    # General compilation options
    cmd += [f"--iree-hal-target-backends={self.args.device}"]

    if self.args.device == "cuda":
      cmd += [f"--iree-hal-cuda-llvm-target-arch={self.args.cuda_arch}"]
    if self.args.split_k_slices != "":
      cmd += [f"--iree-flow-split-matmul-reduction={self.args.split_k_slices}"]
    if self.args.use_mma_sync:
      cmd += [f"--iree-codegen-llvmgpu-use-mma-sync"]
    if self.args.use_wmma:
      cmd += [f"--iree-codegen-llvmgpu-use-wmma"]

    # Compilation options for profiling
    cmd += [
        f"--iree-hal-benchmark-dispatch-repeat-count={benchmark_dispatch_repeat_count}"
    ]

    # Appends print ir options at the end of the command line.
    if self.args.mlir_print_ir_after_all:
      cmd += [f"--mlir-print-ir-after-all"]

    if not os.path.exists(vmfb_file) or self.args.force_compile:
      print(f">> Compilation command for "
            f"{CompilationModeNames[compilation_mode]} : {' '.join(cmd)}")

      compile_log_filename = f"{self.operation_path}/iree_compile_logs.mlir"
      with open(compile_log_filename, "w") as fp:
        subprocess.run(cmd, stderr=fp)

    elif self.args.verbose:
      print("Skipping compilation of operation: " + vmfb_file +
            " since it already exists.")

  def verify(self, configuration):
    """Verifies the  operation with a given configuration."""
    # First compile the operation to a vmfb file.
    self.iree_compile(CompilationMode.Verify)

    # Verify using random data distribution.
    reference_run = self.reference_impl_map[self.operation.operation_kind](\
        self.operation, self.op_reference_cache_path, Distribution.Random, Distribution.Random)

    if not reference_run.is_cached():
      reference_run()

    # Commandline `iree-run-module` for verification.
    cmd = [
        self.iree_run_module_path, f'--module={self.vmfb_verify_file}',
        f'--device={self.args.device}'
    ]

    # Operation-specific verification command-line.
    cmd.append(f'--function={self.operation.name()}_{configuration.name()}')
    for input_file_path in reference_run.get_input_filepaths():
      cmd.append(f'--input=@{input_file_path}')
    for output_file_path in reference_run.get_output_filepaths():
      cmd.append(f'--expected_output=@{output_file_path}')

    # Print the command if verbose.
    if self.args.verbose:
      print(">> Verification command: " + ' '.join(cmd))

    # Launch verification.
    cmd_output = subprocess.check_output(cmd, text=True)

    # Parse the verification output.
    m = re.search(r"\[(?P<verification_result>[a-zA-Z]+)\]", cmd_output)
    if m is None:
      raise Exception(
          "Failed to parse verification output by iree-run-module: " +
          cmd_output)
    verification_result = m.group('verification_result')

    if self.args.verbose or verification_result != "SUCCESS":
      print(cmd_output)

    return verification_result

  def profile(self, configuration):
    """Profiles the matmul operation with a given configuration."""
    # First compile the operation to a vmfb file.
    self.iree_compile(CompilationMode.Profile)

    # Commandline `iree-benchmark-module` for profiling.
    cmd = [
        self.iree_benchmark_module_path, f'--module={self.vmfb_benchmark_file}',
        f'--device={self.args.device}'
    ]

    # Profiling specific flags.
    cmd += [f'--benchmark_repetitions={self.args.benchmark_repetitions}']
    cmd += [f'--batch_size={self.batch_size}']

    # Operation-specific profiling command-line.
    cmd += [f'--function={self.operation.name()}_{configuration.name()}']
    cmd += [f'--input={self.operation.lhs_npy_shape()}']
    cmd += [f'--input={self.operation.rhs_npy_shape()}']

    # Print the command if verbose.
    if self.args.verbose:
      print(">> Profiling command: " + ' '.join(cmd))

    # Launch profiling.
    cmd_output = subprocess.check_output(cmd,
                                         text=True,
                                         stderr=subprocess.STDOUT)

    # Parse the profiling output.
    m = re.search(r"real_time_median\s+(?P<runtime>\d+.\d+)\s+ms", cmd_output)
    if m is None:
      raise Exception("Failed to parse runtime from benchmark result: " +
                      cmd_output)
    runtime_in_ms = float(m.group('runtime'))
    return runtime_in_ms