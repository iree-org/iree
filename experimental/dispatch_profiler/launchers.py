# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from library import *
from matmul import ReferenceMatmulOp
from batch_matmul import ReferenceBatchMatmulOp
from pathlib import Path
import subprocess


class IreeToolsLauncher:
  """Launcher for IREE tools."""

  def __init__(self, args, operation):
    self.operation = operation

    self.generated_path = Path(args.generated_dir, 'generated',
                               args.mlir_dialect)

    self.args = args
    self.benchmark_dispatch_repeat_count = args.batch_size
    self.batch_size = args.batch_size

    # paths to source dispatch mlir, compiled vmfb, and logs.
    self.operation_path = self.generated_path.joinpath(
        OperationKindNames[operation.operation_kind], operation.name())

    self.source_mlir_file = self.operation_path.joinpath(
        operation.name()).with_suffix(".mlir")

    # path to cached numpy refernece input and expected output files.
    self.op_reference_cache_path = Path(args.generated_dir, 'generated',
                                        'reference_cache', operation.name())

    if not self.op_reference_cache_path.exists():
      self.op_reference_cache_path.mkdir(parents=True, exist_ok=True)

    # path to iree-compile tool. (for compiling the input mlir file to vmfb)
    self.iree_compile_path = Path(args.build_dir, 'tools', 'iree-compile')

    # path to iree-benchmark-module tool. (for performance benchmarking and profiling)
    self.iree_benchmark_module_path = Path(args.build_dir, 'tools',
                                           'iree-benchmark-module')

    # path to iree-run-module tool. (for verification)
    self.iree_run_module_path = Path(args.build_dir, 'tools', 'iree-run-module')

    # output vmfb files for verification and profiling.
    vmfb_filename = f"{operation.name()}"

    if operation.operation_kind == OperationKind.SplitkMatmul:
      split_k_suffix = "_".join(
          ['split_k_slice', str(operation.split_k_slices)])
      vmfb_filename = f"{vmfb_filename}_{split_k_suffix}"

    self.vmfb_verify_filepath = self.operation_path.joinpath(
        self.operation.name()).with_name(f"{vmfb_filename}_verify.vmfb")
    self.vmfb_profile_filepath = self.operation_path.joinpath(
        self.operation.name()).with_name(f"{vmfb_filename}_profile.vmfb")

    # reference implementation for the operation_kind.
    self.reference_impl_map = {
        OperationKind.Matmul: ReferenceMatmulOp,
        OperationKind.SplitkMatmul: ReferenceMatmulOp,
        OperationKind.BatchMatmul: ReferenceBatchMatmulOp,
    }

  def iree_compile(self, compilation_mode):
    """Compiles the input mlir file to vmfb file."""

    benchmark_dispatch_repeat_count = self.benchmark_dispatch_repeat_count if compilation_mode == CompilationMode.Profile else 1
    vmfb_filepath = self.vmfb_profile_filepath if compilation_mode == CompilationMode.Profile else self.vmfb_verify_filepath

    # Base iree-compile commandline
    cmd = [
        f'{self.iree_compile_path}',
        f'{self.source_mlir_file}',
        "-o",
        f'{vmfb_filepath}',
    ]

    # General compilation options
    cmd += [f"--iree-hal-target-backends={self.args.device}"]

    if self.args.device == "cuda":
      cmd += [f"--iree-hal-cuda-llvm-target-arch={self.args.cuda_arch}"]
    if self.operation.operation_kind == OperationKind.SplitkMatmul:
      cmd += [
          f"--iree-flow-split-matmul-reduction={self.operation.split_k_slices}"
      ]
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

    if not vmfb_filepath.exists() or self.args.force_compile:
      complie_mode_str = CompilationModeNames[compilation_mode]

      print(f"[Compiling ({complie_mode_str})] {' '.join(cmd)}")

      iree_compile_stdout_filepath = self.operation_path.joinpath(
          'iree_compile_cmd_stdout.mlir')

      with open(iree_compile_stdout_filepath, "w") as fp:
        subprocess.run(cmd, stderr=fp)

    elif self.args.verbose:
      print(
          f"Skipping compilation of operation: {vmfb_filepath} since it already exists."
      )

  def verify(self, configuration):
    """Verifies the operation with a given configuration."""
    # First compile the operation to a vmfb file.
    self.iree_compile(CompilationMode.Verify)

    # Verify using random data distribution.
    reference_run = self.reference_impl_map[self.operation.operation_kind](
        self.operation, self.op_reference_cache_path, Distribution.Random,
        Distribution.Random)

    if not reference_run.is_cached():
      reference_run()

    # Commandline `iree-run-module` for verification.
    cmd = [
        f'{self.iree_run_module_path}', f'--module={self.vmfb_verify_filepath}',
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
      print(f"[Verification] {' '.join(cmd)}")

    # Launch verification.
    cmd_output = subprocess.check_output(cmd, text=True)

    # Save the verification command and the output, only if requested
    # (file writing could slow down the verification).
    if self.args.save_cmds:
      filepath = self.operation_path.joinpath("iree_run_module.stdout")
      with open(filepath, "w") as fp:
        fp.write(f"[Command] $ {' '.join(cmd)}\n")
        fp.write(cmd_output)

    # Parse the verification output.
    m = re.search(r"\[(?P<verification_result>[a-zA-Z]+)\]", cmd_output)
    if m is None:
      raise ValueError(
          f"Failed to parse verification output by iree-run-module: {cmd_output}"
      )
    verification_result = m.group('verification_result')

    if self.args.verbose or verification_result != "SUCCESS":
      print(cmd_output)

    return verification_result

  def profile(self, configuration):
    """Profiles the operation with a given configuration."""
    # First compile the operation to a vmfb file.
    self.iree_compile(CompilationMode.Profile)

    # Commandline `iree-benchmark-module` for profiling.
    cmd = [
        f'{self.iree_benchmark_module_path}',
        f'--module={self.vmfb_profile_filepath}', f'--device={self.args.device}'
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
      print(f"[Profiling] {' '.join(cmd)}")

    # Launch profiling.
    cmd_output = subprocess.check_output(cmd,
                                         text=True,
                                         stderr=subprocess.STDOUT)

    # Save the profiling command and the output, only if requested
    # (file writing could slow down the profiling).
    if self.args.save_cmds:
      filepath = self.operation_path.joinpath("iree_benchmark_module.stdout")
      with open(filepath, "w") as fp:
        fp.write(f"[Command] $ {' '.join(cmd)}\n")
        fp.write(cmd_output)

    # Parse the profiling output.
    m = re.search(r"real_time_median\s+(?P<runtime>\d+.\d+)\s+ms", cmd_output)
    if m is None:
      raise ValueError(
          f"Failed to parse runtime from benchmark result: {cmd_output}")
    runtime_in_ms = float(m.group('runtime'))
    return runtime_in_ms
