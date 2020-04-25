// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  func @scalar_rgn_dispatch_0(%arg0: memref<f32>)
    attributes {iree.dispatch_fn_name = "scalar_rgn_dispatch_0"} {
    %cst = constant dense<1.000000e+00> : tensor<f32>
    //CHECK: %{{.+}} = spv.GLSL.Exp %{{.+}} : f32
    %0 = "xla_hlo.exponential"(%cst) : (tensor<f32>) -> tensor<f32>
    iree.store_output(%0 : tensor<f32>, %arg0 : memref<f32>)
    return
  }
}

// -----

module {
  func @exp(%arg0: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "exp"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: %{{.+}} = spv.GLSL.Exp %{{.+}} : f32
    %2 = "xla_hlo.exponential"(%0) : (tensor<12x42xf32>) -> tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}
