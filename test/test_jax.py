# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import multiprocessing
import re
import subprocess
import sys

parser = argparse.ArgumentParser(prog='test_jax.py',
                                 description='Run jax testsuite hermetically')
parser.add_argument('testfiles', nargs="*")
parser.add_argument('-t', '--timeout', default=20)
parser.add_argument('-l', '--logdir', default="/tmp/jaxtest")
parser.add_argument('-p', '--passing', default=None)
parser.add_argument('-f', '--failing', default=None)
parser.add_argument('-e', '--expected', default=None)

args = parser.parse_args()

PYTEST_CMD = [
    "pytest", "-p", "openxla_pjrt_artifacts",
    f"--openxla-pjrt-artifact-dir={args.logdir}"
]


def get_tests(tests):
  testlist = []
  for test in sorted(tests):
    stdout = subprocess.run(PYTEST_CMD + ["--setup-only", test],
                            capture_output=True)
    testlist += re.findall('::[^ ]*::[^ ]*', str(stdout))
    testlist = [test + func for func in testlist]
  return testlist


def generate_test_commands(tests, timeout=False):
  cmd = ["timeout", f"{args.timeout}"] if timeout else []
  cmd += PYTEST_CMD
  cmds = []
  for test in tests:
    test_cmd = cmd + [test]
    cmds.append(test_cmd)

  return cmds


def exec_test(command):
  result = subprocess.run(command,
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL)
  sys.stdout.write(".")
  sys.stdout.flush()
  return result.returncode


def exec_testsuite(commands):
  returncodes = []
  with multiprocessing.Pool() as p:
    returncodes = p.map(exec_test, commands)
    print("")
  passing = []
  failing = []
  for code, cmd in zip(returncodes, commands):
    testname = " ".join(cmd)
    testname = re.search("[^ /]*::[^ ]*::[^ ]*", testname)[0]

    if code == 0:
      passing.append(testname)
    else:
      failing.append(testname)
  return passing, failing


def write_results(filename, results):
  if (filename is not None):
    with open(filename, 'w') as f:
      for line in results:
        f.write(line + "\n")


def load_results(filename):
  if not filename:
    return []
  expected = []
  with open(filename, 'r') as f:
    for line in f:
      expected.append(line.strip())
  return expected


def compare_results(expected, passing):
  passing = set(passing)
  expected = set(expected)
  new_failures = expected - passing
  new_passing = passing - expected
  return new_passing, new_failures


print("Querying All Tests")
tests = get_tests(args.testfiles)

print("Generating test suite")
test_commands = generate_test_commands(tests, timeout=True)

print(f"Executing {len(test_commands)} tests hermetically")
passing, failing = exec_testsuite(test_commands)
expected = load_results(args.expected)

write_results(args.passing, passing)
write_results(args.failing, failing)

print("Total:", len(test_commands))
print("Passing:", len(passing))
print("Failing:", len(failing))

if expected:
  new_passing, new_failures = compare_results(expected, passing)

  if new_passing:
    print("Newly Passing Tests:")
    for test in new_passing:
      print(" ", test)

  if new_failures:
    print("Newly Failing Tests:")
    for test in new_failures:
      print(" ", test)

  if len(expected) > len(passing):
    exit(1)
