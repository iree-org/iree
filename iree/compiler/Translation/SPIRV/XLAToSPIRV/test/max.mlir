// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

module {
  func @maxf(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes {iree.dispatch_fn_name = "maxf"} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.FMax [[VAL1:%.*]], [[VAL2:%.*]] : f32
    %2 = xla_hlo.maximum %0, %1 : tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    return
  }
}

// -----

module {
  func @maxi(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2 : memref<12x42xi32>)
  attributes {iree.dispatch_fn_name = "maxi"} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.SMax [[VAL1:%.*]], [[VAL2:%.*]] : i32
    %2 = xla_hlo.maximum %0, %1 : tensor<12x42xi32>
    iree.store_output(%2 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    return
  }
}

