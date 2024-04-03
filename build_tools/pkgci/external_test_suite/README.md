# IREE External Test Suite

These files are used to configure test cases from the test suite hosted at
[SHARK-TestSuite/iree_tests](https://github.com/nod-ai/SHARK-TestSuite/tree/main/iree_tests).

Each test case in the test suite includes:

* An input program as a .mlir file
* Inputs and expected outputs, collected into a flagfile

Each configuration file includes:

* Flags for `iree-compile`
* Flags for `iree-run-module`
* Lists of test cases that are expected to fail
