// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule -split-input-file | FileCheck %s

func.func @test_match(%arg0: tensor<250x500xf32>, %arg1: tensor<500x1020xf32>) -> tensor<250x1020xf32> {
  %0 = linalg.init_tensor [250, 1020] : tensor<250x1020xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%1 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
  return %2: tensor<250x1020xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.iree.match ops{["linalg.matmul", "linalg.init_tensor"]} in %arg1 

    //     CHECK: linalg.init_tensor
    // CHECK-NOT: linalg.fill
    //     CHECK: linalg.matmul
    transform.print %0 { name = "Matched Ops" }
  }
}

// The printing of the func comes after the transform.print
//         CHECK: @test_match

// -----

func.func @test_match(%arg0: tensor<250x500xf32>, %arg1: tensor<500x1020xf32>) -> tensor<250x1020xf32> {
  %0 = linalg.init_tensor [250, 1020] : tensor<250x1020xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%1 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
  return %2: tensor<250x1020xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.iree.match interface{LinalgOp} in %arg1 

    // CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<250x1020xf32>) -> tensor<250x1020xf32>
    // CHECK: linalg.matmul ins(%{{.*}}, %{{.*}} : tensor<250x500xf32>, tensor<500x1020xf32>) outs(%{{.*}} : tensor<250x1020xf32>) -> tensor<250x1020xf32>
    transform.print %0 { name = "Matched Interfaces" }
  }
}

// The printing of the func comes after the transform.print
//         CHECK: @test_match
