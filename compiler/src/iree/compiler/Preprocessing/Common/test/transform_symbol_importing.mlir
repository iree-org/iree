// RUN: iree-opt --transform-preload-library='transform-library-paths=%p/external_function_spec.mlir' --transform-interpreter %s | FileCheck %s

module @example {
  // empty
}

// CHECK-LABEL: module @example
//       CHECK:   util.func private @some_external_function(%arg0: tensor<?xf32>) -> tensor<?xf32>
//       CHECK:   util.func public @some_function(%arg0: tensor<?xf32>) -> tensor<?xf32>
//  CHECK-NEXT:     util.return %arg0 : tensor<?xf32>
