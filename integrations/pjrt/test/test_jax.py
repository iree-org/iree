# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import multiprocessing
import os
import random
import re
import subprocess
import sys
import time

from collections import namedtuple
from enum import Enum

parser = argparse.ArgumentParser(prog='test_jax.py',
                                 description='Run jax testsuite hermetically')
parser.add_argument('testfiles', nargs="*")
parser.add_argument('-t', '--timeout', default=60)
parser.add_argument('-l', '--logdir', default="/tmp/jaxtest")
parser.add_argument('-p', '--passing', default=None)
parser.add_argument('-f', '--failing', default=None)
parser.add_argument('-e', '--expected', default=None)
parser.add_argument('-j', '--jobs', default=None)

args = parser.parse_args()

PYTEST_CMD = [
    "pytest", "-p", "openxla_pjrt_artifacts",
    f"--openxla-pjrt-artifact-dir={args.logdir}"
]


def get_test(test):
  print("Fetching from:", test)
  stdout = subprocess.run(PYTEST_CMD + ["--setup-only", test],
                          capture_output=True)
  lst = re.findall('::[^ ]*::[^ ]*', str(stdout))
  return [test + func for func in lst]

def get_tests(tests):
  fulltestlist = []
  with multiprocessing.Pool(os.cpu_count()) as p:
    fulltestlist = p.map(get_test, tests)
  fulltestlist = sorted([i for lst in fulltestlist for i in lst])
  return fulltestlist


def generate_test_commands(tests):
  cmds = []
  for test in tests:
    test_cmd = PYTEST_CMD + [test]
    cmds.append(test_cmd)

  return cmds

TestCase = namedtuple('TestCase', ['test', 'timeout'])
TestResult = Enum('TestResult', ['SUCCESS', 'FAILURE', 'TIMEOUT'])

def exec_test(testcase):
  command, timeout = testcase
  if float(timeout) > 0:
    command = ["timeout", f"{timeout}"] + command

  start = time.perf_counter()
  result = subprocess.run(command, capture_output=True)
  end = time.perf_counter()
  ellapsed = end - start
  timedout = (float(timeout) > 0) and (ellapsed > float(timeout))

  if result.returncode == 0:
    sys.stdout.write(".")
    sys.stdout.flush()
    return TestResult.SUCCESS

  if timedout:
    sys.stdout.write("t")
    sys.stdout.flush()
    return TestResult.TIMEOUT

  sys.stdout.write("f")
  sys.stdout.flush()
  return TestResult.FAILURE


def exec_testsuite(commands, jobs, timeout):
  random.shuffle(commands)
  withTimeout = list(map(lambda x : TestCase(x, timeout), commands))

  results = []
  with multiprocessing.Pool(int(jobs)) as p:
    results = p.map(exec_test, withTimeout)

  passing, timeout, failing = [], [], []
  for result, cmd in zip(results, commands):
    if result == TestResult.SUCCESS:
      passing.append(cmd)

    if result == TestResult.TIMEOUT:
      timeout.append(cmd)

    if result == TestResult.FAILURE:
      failing.append(cmd)
  print("")

  return passing, timeout, failing

def get_testnames(cmd):
  names = []
  for c in cmd:
    testname = " ".join(c)
    testname = re.search("[^ /]*::[^ ]*::[^ ]*", testname)[0]
    names.append(testname)
  return names


def write_results(filename, results):
  if (filename is not None):
    with open(filename, 'w') as f:
      for line in results:
        f.write(line + "\n")


def load_results(filename):
  if not filename or not os.path.isfile(filename):
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
commands = generate_test_commands(tests)

print(f"Executing {len(commands)} tests across {args.jobs} threads with timeout = {args.timeout}")
passing, timeout, failing = exec_testsuite(commands, jobs=args.jobs, timeout=args.timeout)

expected = load_results(args.expected)

# Break into passing vs failing
failing = failing + timeout

# Get the testnames
passing = get_testnames(passing)
failing = get_testnames(failing)

write_results(args.passing, passing)
write_results(args.failing, failing)

print("Total:", len(commands))
print("Passing:", len(passing))
print("Failing:", len(failing))
print("Failing (timed out):", len(timeout))

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
