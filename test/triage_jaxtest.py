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


def check_custom_call(errortxt, _):
  return "mhlo.custom_call" in errortxt


def check_uint(errortxt, _):
  return "uint" in errortxt


def check_degenerate_tensor(_, mlirbc):
  return "tensor<0x" in mlirbc or "x0x" in mlirbc


def check_complex(errortxt, _):
  return "complex<" in errortxt


def check_truncsfbf2(errortxt, _):
  return "__truncsfbf2" in errortxt


def check_scatter_i1(errortxt, _):
  return "'iree_linalg_ext.scatter' op mismatch in argument 0 of region 'i1' and element type of update value 'i8'" in errortxt


def check_dot_i1(_, mlirbc):
  for line in mlirbc.split("\n"):
    has_i1 = re.search("tensor<([0-9]*x)*i1>", line)
    has_dot = re.search("stablehlo.dot", line)
    if has_i1 and has_dot:
      return True
  return False


KnownChecks = {
    "https://github.com/openxla/iree/issues/12410 (custom call)":
        check_custom_call,
    "https://github.com/openxla/iree/issues/12665 (unsigned)   ":
        check_uint,
    "https://github.com/openxla/iree/issues/13347 (0-length)   ":
        check_degenerate_tensor,
    "https://github.com/openxla/iree/issues/12747 (complex)    ":
        check_complex,
    "https://github.com/openxla/iree/issues/13499 (truncsfbf2) ":
        check_truncsfbf2,
    "https://github.com/openxla/iree/issues/13427 (scatter i1) ":
        check_scatter_i1,
    "https://github.com/openxla/iree/issues/13493 (dot i1)     ":
        check_dot_i1,
    "Untriaged":
        lambda _, __: True,
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
    mlirbc_count = len(
        [f for f in os.listdir(f'{args.logdir}/{test}') if "mlirbc" in f])
    mlirbc_name = f'{mlirbc_count - 1}-program.mlirbc'

    with mlir.make_ir_context() as ctx:
      with open(f'{args.logdir}/{test}/{mlirbc_name}', 'rb') as f:
        mlirbc = f.read()
      mlirbc = str(mlir_ir.Module.parse(mlirbc))

    error_mapping[test] = "unknown error"
    for checkname in KnownChecks:
      if KnownChecks[checkname](error, mlirbc):
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
    print(f'{error} : {len(summary[error])}')


failing = filter_to_failures(tests)
mapping = filter_error_mapping(failing)
summary = generate_summary(mapping)
print_summary(summary)
print("Passing:", len(tests) - len(failing))
print("Failing:", len(failing))
