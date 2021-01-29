// RUN: iree-opt %s --iree-codegen-linalg-rewrite-destructive-updates -canonicalize -cse -split-input-file | IreeFileCheck %s

// CHECK-LABEL: func @tile_from_tensor_load
func @tile_from_tensor_load() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %0 = hal.interface.load.tensor @legacy_io::@TENSOR_LHS, offset = %c0
    {operand_result_index = 0 : i32} : tensor<2x3xf32>
  %1 = hal.interface.load.tensor @legacy_io::@TENSOR_RHS, offset = %c0
    {operand_result_index = 1 : i32} : tensor<3x4xf32>
  %2 = hal.interface.load.tensor @legacy_io::@TENSOR_INIT, offset = %c0
    {operand_result_index = 2 : i32} : tensor<2x4xf32>

  %3 = hal.interface.workgroup.id[0] : index
  %4 = hal.interface.workgroup.id[1] : index
  // Test step %{{[0-9a-z]}} { to ensure yield has been folded away.
  //      CHECK: scf.for %[[I:.*]] = {{.*}} step %{{[0-9a-z]+}} {
  //      CHECK:   scf.for %[[J:.*]] = {{.*}} step %{{[0-9a-z]+}} {
  //  CHECK-DAG:     %[[LHS:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_LHS, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32>
  //  CHECK-DAG:     %[[RHS:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_RHS, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [0, %[[J]]], sizes = [3, 1], strides = [1, 1] : tensor<3x1xf32>
  //  CHECK-DAG:     %[[OUT:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_INIT, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], %[[J]]], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32>
  // Compute.
  //      CHECK:     %[[RES:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<1x3xf32>, tensor<3x1xf32>)
  // CHECK-SAME:                       outs(%[[OUT]] : tensor<1x1xf32>)
  // Inplace update (assume only parallel loops have been tiled).
  //      CHECK:     hal.interface.store.tensor.tile %[[RES]], @legacy_io::@ret0, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], %[[J]]], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32>
  %5 = scf.for %arg0 = %4 to %c2 step %c2 iter_args(%arg1 = %2) -> (tensor<2x4xf32>) {
    %6 = scf.for %arg2 = %3 to %c4 step %c4 iter_args(%arg3 = %arg1) -> (tensor<2x4xf32>) {
      %7 = subtensor %0[%arg0, 0] [1, 3] [1, 1] : tensor<2x3xf32> to tensor<1x3xf32>
      %8 = subtensor %1[0, %arg2] [3, 1] [1, 1] : tensor<3x4xf32> to tensor<3x1xf32>
      %9 = subtensor %arg3[%arg0, %arg2] [1, 1] [1, 1] : tensor<2x4xf32> to tensor<1x1xf32>
      %10 = linalg.matmul ins(%7, %8 : tensor<1x3xf32>, tensor<3x1xf32>)
                         outs(%9 : tensor<1x1xf32>)
        -> tensor<1x1xf32>
      // This op is the destructive update we seek to eliminate.
      %11 = subtensor_insert %10 into %arg3[%arg0, %arg2] [%c1, %c1] [%c1, %c1] :
        tensor<1x1xf32> into tensor<2x4xf32>
      scf.yield %11 : tensor<2x4xf32>
    }
    scf.yield %6 : tensor<2x4xf32>
  }

  // Under parallel iterators tiling assumptions, simple forwarding occurs.
  //  CHECK-NOT: hal.interface.load.tensor %
  //  CHECK-NOT: hal.interface.store.tensor %
   // This is the store onto which the destructive update rewrites latches.
  hal.interface.store.tensor %5, @legacy_io::@ret0, offset = %c0
    {operand_result_index = 3 : i32} : tensor<2x4xf32>

  return
}

// -----

#accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL: func @tile_from_pointwise_lhs
func @tile_from_pointwise_lhs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %0 = hal.interface.load.tensor @legacy_io::@TENSOR_INIT, offset = %c0
    {operand_result_index = 0 : i32} : tensor<2x4xf32>
  // This is the `lhs` operand that gets through a pointwise generic and becomes
  // the first input operand of linalg.matmul. After tiling and fusion this is a
  // subtensor of the original `hal.interface.load.tensor`.
  %1 = hal.interface.load.tensor @legacy_io::@TENSOR_LHS, offset = %c0
    {operand_result_index = 1 : i32} : tensor<2x3xf32>
  %2 = hal.interface.load.tensor @legacy_io::@TENSOR_RHS, offset = %c0
    {operand_result_index = 2 : i32} : tensor<3x4xf32>

  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index
  // Test step %{{[0-9a-z]}} { to ensure yield has been folded away.
  //      CHECK: scf.for %[[I:.*]] = {{.*}} step %{{[0-9a-z]+}} {
  //      CHECK:   scf.for %[[J:.*]] = {{.*}} step %{{[0-9a-z]+}} {
  //      CHECK:     %[[LHS:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_LHS, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32>
  //      CHECK:     %[[RHS:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_RHS, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [0, %[[J]]], sizes = [3, 1], strides = [1, 1] : tensor<3x1xf32>
  // Compute.
  //      CHECK:     %[[LHS2:.*]] = linalg.generic {{.*}} ins(%[[LHS]] : tensor<1x3xf32>)
  // TODO: should we reorder this load?
  //      CHECK:     %[[OUT:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_INIT, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], %[[J]]], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32>
  // Compute.
  //      CHECK:     %[[RES:.*]] = linalg.matmul ins(%[[LHS2]], %[[RHS]] : tensor<1x3xf32>, tensor<3x1xf32>)
  // CHECK-SAME:                       outs(%[[OUT]] : tensor<1x1xf32>)
  // Inplace update (assume only parallel loops have been tiled).
  //      CHECK:     hal.interface.store.tensor.tile %[[RES]], @legacy_io::@ret0, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], %[[J]]], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32>
  %cst = constant 1.000000e+00 : f32
  %6 = scf.for %arg0 = %5 to %c2 step %c2 iter_args(%arg1 = %0) -> (tensor<2x4xf32>) {
    %7 = scf.for %arg2 = %4 to %c4 step %c4 iter_args(%arg3 = %arg1) -> (tensor<2x4xf32>) {
      %8 = subtensor %1[%arg0, 0] [1, 3] [1, 1] : tensor<2x3xf32> to tensor<1x3xf32>
      %9 = subtensor %2[0, %arg2] [3, 1] [1, 1] : tensor<3x4xf32> to tensor<3x1xf32>
      %shape = linalg.init_tensor [1, 3] : tensor<1x3xf32>
      %10 = linalg.generic #trait ins(%8 : tensor<1x3xf32>) outs(%shape : tensor<1x3xf32>) {
      ^bb0(%arg4: f32, %s: f32):
        %12 = addf %cst, %arg4 : f32
        linalg.yield %12 : f32
      } -> tensor<1x3xf32>

      %11 = subtensor %arg3[%arg0, %arg2] [1, 1] [1, 1] : tensor<2x4xf32> to tensor<1x1xf32>
      %13 = linalg.matmul ins(%10, %9 : tensor<1x3xf32>, tensor<3x1xf32>)
                         outs(%11 : tensor<1x1xf32>) -> tensor<1x1xf32>

      // This op is the destructive update we seek to eliminate.
      %14 = subtensor_insert %13 into %arg3[%arg0, %arg2] [%c1, %c1] [%c1, %c1] :
        tensor<1x1xf32> into tensor<2x4xf32>
      scf.yield %14 : tensor<2x4xf32>
    }
    scf.yield %7 : tensor<2x4xf32>
  }

  // Under parallel iterators tiling assumptions, simple forwarding occurs.
  //  CHECK-NOT: hal.interface.load.tensor %
  //  CHECK-NOT: hal.interface.store.tensor %
  // This is the store onto which the destructive update rewrites latches.
  hal.interface.store.tensor %6, @legacy_io::@ret0, offset = %c0
    {operand_result_index = 3 : i32} : tensor<2x4xf32>

  return
}

// -----

#accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"]
}

// CHECK-LABEL: func @tile_from_pointwise_outs
func @tile_from_pointwise_outs() {
  %c0 = constant 0 : index
  %c2 = constant 2 : index
  %c4 = constant 4 : index
  %c1 = constant 1 : index
  %cst = constant 1.000000e+00 : f32
  // This is the `out` operand that gets through a pointwise generic and becomes
  // the outs operand of linalg.matmul. After tiling and fusion this is a
  // subtensor of the value produced by linalg.generic.
  %0 = hal.interface.load.tensor @legacy_io::@TENSOR_INIT, offset = %c0
    {operand_result_index = 0 : i32} : tensor<2x4xf32>
  %1 = hal.interface.load.tensor @legacy_io::@TENSOR_LHS, offset = %c0
    {operand_result_index = 1 : i32} : tensor<2x3xf32>
  %2 = hal.interface.load.tensor @legacy_io::@TENSOR_RHS, offset = %c0
    {operand_result_index = 2 : i32} : tensor<3x4xf32>

  // This op is left over and should be DCE'ed but its result is currently used
  // for a destructive update.
  %shape = linalg.init_tensor [2, 4] : tensor<2x4xf32>
  %3 = linalg.generic #trait ins(%0 : tensor<2x4xf32>) outs(%shape : tensor<2x4xf32>) {
  ^bb0(%arg0: f32, %s: f32):
    %t = addf %cst, %arg0 : f32
    linalg.yield %t : f32
  } -> tensor<2x4xf32>

  %4 = hal.interface.workgroup.id[0] : index
  %5 = hal.interface.workgroup.id[1] : index

  // Test step %{{[0-9a-z]}} { to ensure yield has been folded away.
  //      CHECK: scf.for %[[I:.*]] = {{.*}} step %{{[0-9a-z]+}} {
  //      CHECK:   scf.for %[[J:.*]] = {{.*}} step %{{[0-9a-z]+}} {
  //      CHECK:     %[[LHS:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_LHS, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32>
  //      CHECK:     %[[RHS:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_RHS, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [0, %[[J]]], sizes = [3, 1], strides = [1, 1] : tensor<3x1xf32>
  //      CHECK:     %[[OUT:.*]] = hal.interface.load.tensor.tile @legacy_io::@TENSOR_INIT, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], %[[J]]], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32>
  // Compute.
  //      CHECK:     %[[OUT2:.*]] = linalg.generic {{.*}} ins(%[[OUT]] : tensor<1x1xf32>)
  //      CHECK:     %[[RES:.*]] = linalg.matmul ins(%[[LHS]], %[[RHS]] : tensor<1x3xf32>, tensor<3x1xf32>)
  // CHECK-SAME:                       outs(%[[OUT2]] : tensor<1x1xf32>)
  // Inplace update (assume only parallel loops have been tiled).
  //      CHECK:     hal.interface.store.tensor.tile %[[RES]], @legacy_io::@ret0, base_offset = %{{[0-9a-z]*}},
  // CHECK-SAME:       offsets = [%[[I]], %[[J]]], sizes = [%c1, %c1], strides = [%c1, %c1] : tensor<1x1xf32>
  %6 = scf.for %arg0 = %5 to %c2 step %c2 iter_args(%arg1 = %3) -> (tensor<2x4xf32>) {
    %7 = scf.for %arg2 = %4 to %c4 step %c4 iter_args(%arg3 = %arg1) -> (tensor<2x4xf32>) {
      %8 = subtensor %1[%arg0, 0] [1, 3] [1, 1] : tensor<2x3xf32> to tensor<1x3xf32>
      %9 = subtensor %2[0, %arg2] [3, 1] [1, 1] : tensor<3x4xf32> to tensor<3x1xf32>
      %10 = subtensor %0[%arg0, %arg2] [1, 1] [1, 1] : tensor<2x4xf32> to tensor<1x1xf32>
      %11 = linalg.generic #trait ins(%10 : tensor<1x1xf32>) outs(%10 : tensor<1x1xf32>) {
      ^bb0(%arg4: f32, %s: f32):
        %12 = addf %cst, %arg4 : f32
        linalg.yield %12 : f32
      } -> tensor<1x1xf32>
      %13 = linalg.matmul ins(%8, %9 : tensor<1x3xf32>, tensor<3x1xf32>)
                         outs(%11 : tensor<1x1xf32>) -> tensor<1x1xf32>
      // This op is the destructive update we seek to eliminate.
      %14 = subtensor_insert %13 into %arg3[%arg0, %arg2] [%c1, %c1] [%c1, %c1] :
        tensor<1x1xf32> into tensor<2x4xf32>
      scf.yield %14 : tensor<2x4xf32>
    }
    scf.yield %7 : tensor<2x4xf32>
  }

  // Under parallel iterators tiling assumptions, simple forwarding occurs.
  //  CHECK-NOT: hal.interface.load.tensor %
  //  CHECK-NOT: hal.interface.store.tensor %
  // This is the store onto which the destructive update rewrites latches.
  hal.interface.store.tensor %6, @legacy_io::@ret0, offset = %c0
    {operand_result_index = 3 : i32} : tensor<2x4xf32>

  return
}
