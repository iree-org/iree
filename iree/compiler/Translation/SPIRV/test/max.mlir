// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | FileCheck %s

module {
  func @maxf(%arg0: memref<12x42xf32>, %arg1: memref<12x42xf32>, %arg2 : memref<12x42xf32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xf32>) : tensor<12x42xf32>
    %1 = iree.load_input(%arg1 : memref<12x42xf32>) : tensor<12x42xf32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.FMax [[VAL1:%.*]], [[VAL2:%.*]] : f32
    %2 = xla_hlo.max %0, %1 : tensor<12x42xf32>
    iree.store_output(%2 : tensor<12x42xf32>, %arg2 : memref<12x42xf32>)
    iree.return
  }
}

// -----

module {
  func @maxi(%arg0: memref<12x42xi32>, %arg1: memref<12x42xi32>, %arg2 : memref<12x42xi32>)
  attributes  {iree.executable.export, iree.executable.workload = dense<[42, 12, 1]> : tensor<3xi32>, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<12x42xi32>) : tensor<12x42xi32>
    %1 = iree.load_input(%arg1 : memref<12x42xi32>) : tensor<12x42xi32>
    //CHECK: [[COMPARE:%.*]] = spv.GLSL.SMax [[VAL1:%.*]], [[VAL2:%.*]] : i32
    %2 = xla_hlo.max %0, %1 : tensor<12x42xi32>
    iree.store_output(%2 : tensor<12x42xi32>, %arg2 : memref<12x42xi32>)
    iree.return
  }
}

