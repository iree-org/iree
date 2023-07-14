# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import jaxlib.mlir.ir as mlir_ir
import jax._src.interpreters.mlir as mlir
import multiprocessing
import os
import re

parser = argparse.ArgumentParser(prog='triage_jaxtest.py',
                                 description='Triage the jax tests')
parser.add_argument('-l', '--logdir', default="/tmp/jaxtest")
parser.add_argument('-d', '--delete', default=False)
parser.add_argument('-j', '--jobs', default=None)
args = parser.parse_args()

tests = set(os.listdir(args.logdir))


def filter_to_failures(tests):
  failures = list()
  for test in tests:
    files = os.listdir(f"{args.logdir}/{test}")
    if "error.txt" in files or "CRASH_MARKER" in files:
      failures.append(test)
  failures = sorted(failures)
  return failures


def check_custom_call(errortxt, mlirbc, __):
  return "stablehlo.custom_call" in errortxt or "stablehlo.custom_call" in mlirbc


def check_load_ui(errortxt, _, __):
  return "'flow.tensor.load' op result #0 must be index or signless integer or floating-point or complex-type or vector of any type values, but got 'ui32'" in errortxt


def check_splat_ui(errortxt, _, __):
  return "'flow.tensor.splat' op failed to verify that value type matches element type of result" in errortxt and "xui" in errortxt


def check_degenerate_tensor(_, mlirbc, __):
  return "tensor<0x" in mlirbc or "x0x" in mlirbc


def check_topk_bf16(_, mlirbc, __):
  return "bf16" in mlirbc and "hlo.top_k" in mlirbc


def check_cross_replica(errortxt, _, __):
  return "cross-replica" in errortxt


def check_collective(errortxt, _, __):
  return "stablehlo.collective" in errortxt or "UNIMPLEMENTED; collectives not implemented" in errortxt


def check_sort_shape(errortxt, _, __):
  return "'iree_linalg_ext.sort' op expected operand 1 to have same shape as other operands" in errortxt


def check_reverse_i1(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    if "stablehlo.reverse" in line and "xui" in line:
      return True
  return False


def check_complex(errortxt, mlirbc, __):
  return "complex<" in errortxt or "complex<" in mlirbc


def check_timeout(errortxt, _, __):
  return "jaxlib.xla_extension.XlaRuntimeError: ABORTED: ABORTED" in errortxt


def check_rng_bit_i8(_, mlirbc, __):
  lines = mlirbc.split("\n")
  for line in lines:
    if "stablehlo.rng_bit_generator" in line and "i8" in line:
      return True
  return False


def check_min_max_f16(errortxt, _, __):
  if "undefined symbol: fminf" in errortxt:
    return True
  lines = errortxt.split("\n")
  for line in lines:
    has_fmax = "llvm.intr.vector.reduce.fmax" in line
    has_fmin = "llvm.intr.vector.reduce.fmin" in line
    has_f16 = "f16" in line
    if (has_fmax or has_fmin) and has_f16:
      return True
  return False


def check_scatter_ui(errortxt, _, __):
  lines = errortxt.split("\n")
  for line in lines:
    has_scatter = "iree_linalg_ext.scatter" in line
    has_operand = "expected type of `outs` operand #0" in line
    has_type = "xui" in line
    if has_scatter and has_operand and has_type:
      return True
  return False


def check_bitcast_bf16(errortxt, _, __):
  return "bf16" in errortxt and "`arith.bitcast` op operand type" in errortxt


def check_constant_bf16(errortxt, _, __):
  return "FloatAttr does not match expected type of the constant" in errortxt


def check_triangular_solve(errortxt, _, __):
  return "stablehlo.triangular_solve" in errortxt


def check_cholesky(errortxt, _, __):
  return "stablehlo.cholesky" in errortxt


def check_fft(_, mlirbc, __):
  return "stablehlo.fft" in mlirbc


def check_schedule_allocation(errortxt, _, __):
  return "Pipeline failed while executing [`ScheduleAllocation`" in errortxt


def check_dot_i1(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    has_i1 = re.search("tensor<([0-9]*x)*i1>", line)
    has_dot = re.search("stablehlo.dot", line)
    if has_i1 and has_dot:
      return True
  return False


def check_vectorize(errortxt, _, __):
  return "arith.truncf' op operand #0 must be floating-point-like, but got 'vector<f32>" in errortxt


def check_roundeven(errortxt, _, __):
  return "roundeven" in errortxt


def check_numerical(errortxt, _, __):
  return "Mismatched elements" in errortxt


def check_compilation(errortxt, _, __):
  return "iree/integrations/pjrt/common/api_impl.cc:1085" in errortxt


def check_scatter(errortxt, _, __):
  return "'iree_linalg_ext.scatter' op mismatch in shape of indices and update value at dim#0" in errortxt


def check_shape_cast(errortxt, _, __):
  return "'vector.shape_cast' op source/result number of elements must match" in errortxt


def check_scatter_crash(_, mlirbc, runtime_crash):
  return "stablehlo.scatter" in mlirbc and runtime_crash


def check_eigen_decomposition(errortxt, _, __):
  return "Nonsymmetric eigendecomposition is only implemented on the CPU backend" in errortxt


def check_jax_unimplemented(errortxt, _, __):
  return "NotImplementedError: MLIR translation rule for primitive" in errortxt


def check_serialize_exe(errortxt, _, __):
  return "UNIMPLEMENTED; PJRT_Executable_Serialize" in errortxt


def check_optimized_prgrm(errortxt, _, __):
  return "UNIMPLEMENTED; PJRT_Executable_OptimizedProgram" in errortxt


def check_optimized_program(errortxt, _, __):
  return "UNIMPLEMENTED; PJRT_Executable_OptimizedProgram" in errortxt


def check_donation(errortxt, _, __):
  return "Donation is not implemented for iree_cpu" in errortxt


def check_semaphore_overload(errortxt, _, __):
  return "OUT_OF_RANGE; semaphore values must be monotonically increasing;" in errortxt


def check_python_callback(errortxt, _, __):
  return "ValueError: `EmitPythonCallback` not supported" in errortxt


def check_complex_convolution(errortxt, mlirbc, __):
  if "failed to legalize operation 'complex.constant'" in errortxt:
    return True

  for line in mlirbc.split("\n"):
    has_i1 = re.search("tensor<([0-9]*x)*complex<f[0-9]*>>", line)
    has_conv = re.search("stablehlo.convolution", line)
    has_dot = re.search("stablehlo.dot", line)
    if has_i1 and (has_conv or has_dot):
      return True
  return False


def check_subspan(errortxt, _, __):
  return "failed to legalize operation 'hal.interface.binding.subspan' that was explicitly marked illegal" in errortxt


def check_from_tensor(errortxt, _, __):
  return "error: 'tensor.from_elements' op unhandled tensor operation" in errortxt


def check_unknown_backend(errortxt, _, __):
  return "RuntimeError: Unknown backend" in errortxt


def check_unsigned_topk(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    if "xui" in line and "chlo.top_k" in line:
      return True
  return False


def check_runtime_crash(__, _, runtime_crash):
  return runtime_crash


def check_aborted(errortxt, _, __):
  return "ABORTED" in errortxt


def check_bounds_indexing(errortxt, _, __):
  return "out-of-bounds indexing for array of shape" in errortxt


def check_nan_correctness(errortxt, _, __):
  return "nan location mismatch" in errortxt


def check_pointer_mismatch(errortxt, _, __):
  return "unsafe_buffer_pointer()" in errortxt


def check_select_and_scatter(errortxt, _, __):
  return "failed to legalize operation 'stablehlo.select_and_scatter'" in errortxt


def check_degenerate_scatter(errortxt, _, __):
  return "'iree_linalg_ext.scatter' op operand #2 must be ranked tensor or memref of any type values" in errortxt


def check_cost_analysis(errortxt, _, __):
  return "cost_analysis()" in errortxt


def check_invalid_option(errortxt, _, __):
  return "No such compile option: 'invalid_key'" in errortxt


def check_inf_mismatch(errortxt, _, __):
  return "inf location mismatch" in errortxt


def check_shape_assertion(errortxt, _, __):
  for line in errortxt.split("\n"):
    if "assertEqual" in line and ".shape" in line:
      return True
  return False


def check_vector_contract(errortxt, _, __):
  return "'vector.contract' op failed to verify that lhs and rhs have same element type" in errortxt


def check_subbyte_read(errortxt, _, __):
  return "opaque and sub-byte aligned element types cannot be indexed" in errortxt


def check_buffer_usage(errortxt, _, __):
  return "requested buffer usage is not supported" in errortxt or "tensor requested usage was not specified when the buffer" in errortxt or "PERMISSION_DENIED; requested usage was not specified when the buffer was allocated; buffer allows DISPATCH_INDIRECT_PARAMS" in errortxt


def check_subbyte_singleton(errortxt, _, __):
  return "does not have integral number of total bytes" in errortxt


def check_max_arg(errortxt, _, __):
  return "max() arg is an empty sequence" in errortxt


def check_double_support(errortxt, _, __):
  return "expected f32 (21000020) but have f64 (21000040)" in errortxt


def check_stablehlo_degenerate(_, mlirbc, __):
  for line in mlirbc.split("\n"):
    if "stablehlo" in line and ("x0x" in line or "<0x" in line):
      return True
  return False


def check_stablehlo_allreduce(errortxt, _, __):
  return "failed to legalize operation 'stablehlo.all_reduce'" in errortxt


def check_dot_shape(errortxt, _, __):
  for line in errortxt.split("\n"):
    if "error: inferred shape" in line and "is incompatible with return type of operation " in line:
      return True
  return False


KnownChecks = {
    "https://github.com/openxla/iree/issues/14255 (detensoring)":
        check_from_tensor,
    "https://github.com/openxla/iree/issues/????? (unknown)":
        check_jax_unimplemented,
    "https://github.com/openxla/iree/issues/13726 (collective)":
        check_collective,
    "https://github.com/openxla/iree/issues/12410 (custom call)":
        check_custom_call,
    "https://github.com/openxla/iree/issues/11018 (triangle)":
        check_triangular_solve,
    "https://github.com/openxla/iree/issues/12263 (fft)":
        check_fft,
    "https://github.com/openxla/iree/issues/14072 (complex convolution)":
        check_complex_convolution,
    "https://github.com/openxla/iree/issues/10816 (cholesky)":
        check_cholesky,
    "https://github.com/openxla/iree/issues/11761 (rng bit gen i8)":
        check_rng_bit_i8,
    "https://github.com/openxla/iree/issues/????? (eigen decomp)":
        check_eigen_decomposition,
    "https://github.com/openxla/iree/issues/13579 (scatter ui)":
        check_scatter_ui,
    "https://github.com/openxla/iree/issues/13725 (cross repl)":
        check_cross_replica,
    "https://github.com/openxla/iree/issues/13493 (dot i1)":
        check_dot_i1,
    "https://github.com/openxla/iree/issues/13522 (roundeven)":
        check_roundeven,
    "https://github.com/openxla/iree/issues/13577 (max/min f16)":
        check_min_max_f16,
    "https://github.com/openxla/iree/issues/13523 (scatter)":
        check_scatter,
    "https://github.com/openxla/iree/issues/13580 (scatter crash)":
        check_scatter_crash,
    "https://github.com/openxla/iree/issues/14079 (shape_cast)":
        check_shape_cast,
    "https://github.com/openxla/iree/issues/????? (optimized prgrm)":
        check_optimized_program,
    "https://github.com/openxla/iree/issues/????? (donation)":
        check_donation,
    "https://github.com/openxla/iree/issues/????? (python callback)":
        check_python_callback,
    "https://github.com/openxla/iree/issues/????? (subspan)":
        check_subspan,
    "https://github.com/openxla/iree/issues/14098 (unsigned topk)":
        check_unsigned_topk,
    "https://github.com/openxla/iree/issues/????? (bounds indexing)":
        check_bounds_indexing,
    "https://github.com/openxla/iree/issues/????? (nan correctness)":
        check_nan_correctness,
    "https://github.com/openxla/iree/issues/????? (pointer mismatch)":
        check_pointer_mismatch,
    "https://github.com/openxla/iree/issues/10841 (select and scatter)":
        check_select_and_scatter,
    "https://github.com/openxla/iree/issues/????? (degenerate scatter)":
        check_degenerate_scatter,
    "https://github.com/openxla/iree/issues/????? (cost analysis)":
        check_cost_analysis,
    "https://github.com/openxla/iree/issues/????? (invalid option)":
        check_invalid_option,
    "https://github.com/openxla/iree/issues/????? (inf mismatch)":
        check_inf_mismatch,
    "https://github.com/openxla/iree/issues/????? (shape assertion)":
        check_shape_assertion,
    "https://github.com/openxla/iree/issues/????? (vector contract)":
        check_vector_contract,
    "https://github.com/openxla/iree/issues/????? (subbyte indexed)":
        check_subbyte_read,
    "https://github.com/openxla/iree/issues/????? (buffer usage)":
        check_buffer_usage,
    "https://github.com/openxla/iree/issues/????? (subbyte singleton)":
        check_subbyte_singleton,
    "https://github.com/openxla/iree/issues/????? (max arg)":
        check_max_arg,
    "https://github.com/openxla/iree/issues/????? (double support)":
        check_double_support,
    "https://github.com/openxla/iree/issues/????? (zero extent)":
        check_stablehlo_degenerate,
    "https://github.com/openxla/iree/issues/????? (all reduce)":
        check_stablehlo_allreduce,
    "https://github.com/openxla/iree/issues/????? (stablehlo dot_general)":
        check_dot_shape,
    "(unknown backend)":
        check_unknown_backend,
    "(semaphore)":
        check_semaphore_overload,
    "Aborted (possible timeout)":
        check_aborted,
    "Runtime Crash":
        check_runtime_crash,
    "Compilation Failure":
        check_compilation,
    "Numerical Failures":
        check_numerical,
    "Untriaged":
        lambda _, __, ___: True,
}


def triage_test(test):
  files = sorted(os.listdir(f'{args.logdir}/{test}'))
  # Load the error.txt if it is available.
  error = ""
  if "error.txt" in files:
    with open(f'{args.logdir}/{test}/error.txt') as f:
      error = "".join(f.readlines())

  # Load the last bytecode file that was attempted to be compiled:
  mlirbc_count = len([f for f in files if "mlirbc" in f])
  mlirbc_name = f'{mlirbc_count - 1}-program.mlirbc'
  vmfb_name = f'{mlirbc_count - 1}-program.vmfb'

  runtime_crash = "CRASH_MARKER" in files

  mlirbc = ""
  if mlirbc_count > 0:
    with mlir.make_ir_context() as ctx:
      with open(f'{args.logdir}/{test}/{mlirbc_name}', 'rb') as f:
        mlirbc = f.read()
      mlirbc = str(mlir_ir.Module.parse(mlirbc))

  for checkname in KnownChecks:
    if KnownChecks[checkname](error, mlirbc, runtime_crash):
      return checkname

  return "unknown error"


def filter_error_mapping(tests):
  error_mapping = {}
  with multiprocessing.Pool(int(args.jobs) if args.jobs else args.jobs) as p:
    results = p.map(triage_test, tests)

  for test, result in zip(tests, results):
    error_mapping[test] = result
  return error_mapping


def generate_summary(mapping):
  summary = {}
  for err in KnownChecks.keys():
    summary[err] = []
  for test in mapping:
    summary[mapping[test]].append(test)
  return summary


def print_summary(summary):
  maxlen = 0
  for error in summary:
    maxlen = max(len(error), maxlen)
  for error in summary:
    print(f'{error:<{maxlen}} : {len(summary[error])}')

  passstr = "Passing"
  failstr = "Failing"
  print(f'{passstr:<{maxlen}} : {len(tests) - len(failing)}')
  print(f'{failstr:<{maxlen}} : {len(failing)}')


failing = filter_to_failures(tests)
mapping = filter_error_mapping(failing)
summary = generate_summary(mapping)
print_summary(summary)
