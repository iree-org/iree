// RUN: iree-opt -split-input-file -iree-vmla-pre-conversion-lowering -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

func private @sort1D(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-DAG: [[C16:%.+]] = constant 16 : index
  // CHECK-DAG: [[RS:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4]>
  // CHECK-DAG: [[BL:%.+]] = vmla.buffer.alloc byte_length = [[C16]] : !vmla.buffer
  // CHECK-DAG: vmla.sort %arg0([[RS]] : !shapex.ranked_shape<[4]>), out [[BL]] : f32
  // CHECK-DAG: [[BUF:%.+]] = vmla.buffer.alloc byte_length = [[C16]] : !vmla.buffer
  // CHECK-DAG: vmla.gather %arg0([[RS]] : !shapex.ranked_shape<[4]>), [[BL]]([[RS]] : !shapex.ranked_shape<[4]>), out [[BUF]]([[RS]] : !shapex.ranked_shape<[4]>) {batch_dims = 0 : i64, dim = 0 : i64} : f32
  %sort = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %compare = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 0 : i64, is_stable = false} : (tensor<4xf32>) -> tensor<4xf32>

  // CHECK: return [[BUF]] : !vmla.buffer
  return %sort : tensor<4xf32>
}


// CHECK-LABEL: func private @sort2D
func private @sort2D(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-DAG: [[C64:%.+]] = constant 64 : index
  // CHECK-DAG: [[RS:%.+]] = shapex.const_ranked_shape : !shapex.ranked_shape<[4,4]>
  // CHECK-DAG: [[BL:%.+]] = vmla.buffer.alloc byte_length = [[C64]] : !vmla.buffer
  // CHECK-DAG: vmla.sort %arg0([[RS]] : !shapex.ranked_shape<[4,4]>), out [[BL]] : f32
  // CHECK-DAG: [[BUF:%.+]] = vmla.buffer.alloc byte_length = [[C64]] : !vmla.buffer
  // CHECK-DAG: vmla.gather %arg0([[RS]] : !shapex.ranked_shape<[4,4]>), [[BL]]([[RS]] : !shapex.ranked_shape<[4,4]>), out [[BUF]]([[RS]] : !shapex.ranked_shape<[4,4]>) {batch_dims = 1 : i64, dim = 1 : i64} : f32
  %sort = "mhlo.sort"(%arg0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
    %compare = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "GT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%compare) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>

  // CHECK: return [[BUF]] : !vmla.buffer
  return %sort : tensor<4x4xf32>
}
