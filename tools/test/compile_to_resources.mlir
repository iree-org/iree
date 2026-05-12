// RUN: iree-compile --compile-to=input %s | FileCheck %s

// CHECK: arith.constant dense_resource<resource_i32> : tensor<4xi32>
// CHECK: {-#
// CHECK-NEXT:   dialect_resources: {
// CHECK-NEXT:     builtin: {
// CHECK-NEXT:       resource_i32: "0x4000000001000000020000000300000004000000"
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: #-}

func.func @resource_constant() -> tensor<4xi32> {
  %cst = arith.constant dense_resource<resource_i32> : tensor<4xi32>
  return %cst : tensor<4xi32>
}

{-#
  dialect_resources: {
    builtin: {
      resource_i32: "0x4000000001000000020000000300000004000000"
    }
  }
#-}
