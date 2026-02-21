// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: @simple_case
util.func @simple_case() {
  util.return
}

// -----

// CHECK-LABEL: @with_case_run
// RUN: iree-opt %s --canonicalize | FileCheck %s --check-prefix=CANON
util.func @with_case_run() {
  util.return
}

// -----

// CHECK-LABEL: @long_pipeline
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(cse,canonicalize,symbol-dce))' %s | FileCheck %s
util.func @long_pipeline() {
  util.return
}

// -----

// CHECK-LABEL: @verify_diagnostics
// RUN: iree-opt %s --verify-diagnostics
util.func @verify_diagnostics() {
  util.return
}

// -----

// CHECK-LABEL: @multiple_check_prefixes
// RUN: iree-opt %s | FileCheck %s --check-prefixes=CHECK,EXTRA
util.func @multiple_check_prefixes() {
  util.return
}

// -----

// CHECK-LABEL: @with_argument
util.func @with_argument(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  util.return %arg0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @pipeline_chain
// RUN: iree-opt %s | iree-opt | FileCheck %s
util.func @pipeline_chain() {
  util.return
}

// -----

// CHECK-LABEL: @environment_variable
// RUN: IREE_TEST_VAR=value iree-opt %s | FileCheck %s
util.func @environment_variable() {
  util.return
}
