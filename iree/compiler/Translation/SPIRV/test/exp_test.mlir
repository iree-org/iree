// RUN: iree-opt -split-input-file -convert-iree-to-spirv -simplify-spirv-affine-exprs=false -verify-diagnostics -o - %s | FileCheck %s

module {
  func @scalar_rgn_dispatch_0(%arg0: memref<f32>)
    attributes  {iree.executable.export, iree.executable.workload = dense<1> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %cst = constant dense<1.000000e+00> : tensor<f32>
    //CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %0 = "xla_hlo.exp"(%cst) : (tensor<f32>) -> tensor<f32>
    iree.store_output(%0 : tensor<f32>, %arg0 : memref<f32>)
    iree.return
  }
}
