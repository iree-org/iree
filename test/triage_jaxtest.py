# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import jaxlib.mlir.ir as mlir_ir
import jax._src.interpreters.mlir as mlir
import os
import re

parser = argparse.ArgumentParser(prog='triage_jaxtest.py',
                                 description='Triage the jax tests')
parser.add_argument('-l', '--logdir', default="/tmp/jaxtest")
parser.add_argument('-d', '--delete', default=False)
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


def check_custom_call(errortxt, _, __):
  return "mhlo.custom_call" in errortxt

def check_load_ui(errortxt, _, __):
  return "'flow.tensor.load' op result #0 must be index or signless integer or floating-point or complex-type or vector of any type values, but got 'ui32'" in errortxt

def check_degenerate_tensor(_, mlirbc, __):
  return "tensor<0x" in mlirbc or "x0x" in mlirbc


def check_complex(errortxt, _, __):
  return "complex<" in errortxt

def check_min_max_f16(errortxt, _, __):
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

def check_truncsfbf2(errortxt, _, __):
  return "__truncsfbf2" in errortxt


def check_scatter_i1(errortxt, _, __):
  return "'iree_linalg_ext.scatter' op mismatch in argument 0 of region 'i1' and element type of update value 'i8'" in errortxt


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

def check_scatter_crash(_, mlirbc, runtime_crash):
  return "stablehlo.scatter" in mlirbc and runtime_crash

def check_runtime_crash(errortxt, _, runtime_crash):
  return runtime_crash

KnownChecks = {
    "https://github.com/openxla/iree/issues/12410 (custom call)":
        check_custom_call,
    "https://github.com/openxla/iree/issues/13576 (flow.load ui)":
        check_load_ui,
    "https://github.com/openxla/iree/issues/13347 (0-length)":
        check_degenerate_tensor,
    "https://github.com/openxla/iree/issues/13544 (complex)":
        check_complex,
    "https://github.com/openxla/iree/issues/13499 (truncsfbf2)":
        check_truncsfbf2,
    "https://github.com/openxla/iree/issues/13427 (scatter i1)":
        check_scatter_i1,
    "https://github.com/openxla/iree/issues/13493 (dot i1)":
        check_dot_i1,
    "https://github.com/openxla/iree/issues/13507 (vectorize)":
        check_vectorize,
    "https://github.com/openxla/iree/issues/13522 (roundeven)":
        check_roundeven,
    "https://github.com/openxla/iree/issues/13579 (scatter ui)":
        check_scatter_ui,
    "https://github.com/openxla/iree/issues/13577 (max/min f16)":
        check_min_max_f16,
    "https://github.com/openxla/iree/issues/13523 (scatter)":
        check_scatter,
    "https://github.com/openxla/iree/issues/13580 (scatter crash)":
        check_scatter_crash,
    "Runtime Crash":
        check_runtime_crash,
    "Compilation Failure":
        check_compilation,
    "Numerical Failures":
        check_numerical,
    "Untriaged":
        lambda _, __, ___: True,
}


def filter_error_mapping(tests):
  error_mapping = {}
  for test in tests:
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

    runtime_crash = "CRASH_MARKER" in files and vmfb_name in files

    with mlir.make_ir_context() as ctx:
      with open(f'{args.logdir}/{test}/{mlirbc_name}', 'rb') as f:
        mlirbc = f.read()
      mlirbc = str(mlir_ir.Module.parse(mlirbc))

    error_mapping[test] = "unknown error"
    for checkname in KnownChecks:
      if KnownChecks[checkname](error, mlirbc, runtime_crash):
        error_mapping[test] = checkname
        break
  return error_mapping


def generate_summary(mapping):
  summary = {}
  for err in KnownChecks.keys():
    summary[err] = []
  for test in mapping:
    summary[mapping[test]].append(test)
  return summary


def print_summary(summary):
  for error in summary:
    print(f'{error:<60} : {len(summary[error])}')


failing = filter_to_failures(tests)
mapping = filter_error_mapping(failing)
summary = generate_summary(mapping)
print_summary(summary)
print("{:<60} : {}".format("Passing", len(tests) - len(failing)))
print("{:<60} : {}".format("Failing", len(failing)))
