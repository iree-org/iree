// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  func @scalar_rgn_dispatch_0(%arg0: memref<f32>)
    attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
    %cst = constant dense<1.000000e+00> : tensor<f32>
    //CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %0 = "xla_hlo.exp"(%cst) : (tensor<f32>) -> tensor<f32>
    iree.store_output(%0 : tensor<f32>, %arg0 : memref<f32>)
    return
  }
}

// -----

module {
  func @exp(%arg0: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %2 = "xla_hlo.exp"(%0) : (tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}
