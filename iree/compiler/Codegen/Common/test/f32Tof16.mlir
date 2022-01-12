// RUN: iree-opt -split-input-file -iree-convert-f32-to-f16 %s | FileCheck %s

//       CHECK: util.global {{.*}} : tensor<4xf16>
// CHECK-LABEL: func @simple_f32() -> tensor<4xf16>
//  CHECK-NEXT: %{{.*}} = util.global.address @__global : !util.ptr<tensor<4xf16>>
//  CHECK-NEXT: %{{.*}} = util.global.load.indirect %{{.*}} : !util.ptr<tensor<4xf16>> -> tensor<4xf16>
//  CHECK-NEXT: return %{{.*}} : tensor<4xf16>
module {
  util.global private @"__global" = dense<"0x000020410000A040000020410000A040"> : tensor<4xf32>
  func @simple_f32() -> (tensor<4xf32>) {
    %0 = util.global.address @"__global" : !util.ptr<tensor<4xf32>>
    %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}

// -----

// CHECK: util.global
// CHECK-NOT: f32
// CHECK-LABEL: func @nested_region_f32()
// CHECK-NOT: f32
// CHECK: return %{{.*}} : tensor<4xf16>
module {
  util.global private @"__global" = dense<"0x000020410000A040000020410000A040"> : tensor<4xf32>
  func @nested_region_f32() -> (tensor<4xf32>) {
    %0 = util.global.address @"__global" : !util.ptr<tensor<4xf32>>
    %1 = util.global.load.indirect %0 : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
    %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4x4xf32>
    %4 = mhlo.constant dense<0xFF800000> : tensor<f32>
    %3 = "mhlo.reduce"(%2, %4) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %5393 = mhlo.maximum %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%5393) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<f32>) -> tensor<4xf32>
    return %3 : tensor<4xf32>
  }
}
