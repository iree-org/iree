// RUN: iree-opt %s | FileCheck %s --dump-input=fail

// CHECK-LABEL: @dispatch_entry
func @dispatch_entry(%arg0: memref<4x2xf32>, %arg1: memref<4x2xf32>) {
  // CHECK-NEXT: %0 = iree.load_input(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  %0 = iree.load_input(%arg0 : memref<4x2xf32>) : tensor<4x2xf32>
  // CHECK-NEXT: iree.store_output(%0 : tensor<4x2xf32>, %arg1 : memref<4x2xf32>)
  iree.store_output(%0 : tensor<4x2xf32>, %arg1 : memref<4x2xf32>)
  return
}
