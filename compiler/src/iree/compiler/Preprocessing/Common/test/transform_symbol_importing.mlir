// RUN: iree-opt --pass-pipeline='builtin.module(iree-preprocessing-transform-interpreter{transform-spec-path=%p/external_function_spec.mlir})' %s | FileCheck %s

module @example {
  // empty
}

// CHECK-LABEL: module @example
//       CHECK:   func.func private @some_external_function(tensor<?xf32>) -> tensor<?xf32>
//       CHECK:   func.func @some_function(%arg0: tensor<?xf32>) -> tensor<?xf32>
//  CHECK-NEXT:     return %arg0 : tensor<?xf32>
