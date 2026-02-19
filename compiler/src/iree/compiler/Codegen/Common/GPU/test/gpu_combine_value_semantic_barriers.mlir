// RUN: iree-opt --iree-codegen-gpu-combine-value-semantic-barriers %s --split-input-file | FileCheck %s

// Since the pass only rearranges the order of instructions, we only check the
// number of value_barriers.

func.func @tensor_barrier(%write: vector<8xf16>, %input: tensor<8xf16>, %input2 : tensor<16xf16>) -> (vector<8xf16>, vector<8xf16>) {
  %c0 = arith.constant 0 : index
  %cv0 = arith.constant 0.0 : f16

  %wait1 = vector.transfer_write %write, %input[%c0] : vector<8xf16>, tensor<8xf16>
  %synced1 = iree_gpu.value_barrier %wait1 : tensor<8xf16>
  %out = vector.transfer_read %synced1[%c0], %cv0 : tensor<8xf16>, vector<8xf16>

  %wait2 = vector.transfer_write %write, %input2[%c0] : vector<8xf16>, tensor<16xf16>
  %synced2 = iree_gpu.value_barrier %wait2 : tensor<16xf16>
  %out2 = vector.transfer_read %synced2[%c0], %cv0 : tensor<16xf16>, vector<8xf16>

  return %out, %out2 : vector<8xf16>, vector<8xf16>
}

// There should be only 1 value_barrier left

// CHECK-LABEL: func.func @tensor_barrier
// CHECK: value_barrier
// CHECK-NOT: value_barrier

// -----

func.func @vector_barrier(%write: vector<8xf16>, %write2: vector<8xf16>) -> vector<8xf16> {
  %synced  =  iree_gpu.value_barrier %write  : vector<8xf16>
  %synced2 =  iree_gpu.value_barrier %write2 : vector<8xf16>
  %add     =  arith.addf %synced, %synced2   : vector<8xf16>
  return %add : vector<8xf16>
}

// There should be only 1 value_barrier left

// CHECK-LABEL: func.func @vector_barrier
// CHECK: value_barrier
// CHECK-NOT: value_barrier

// -----

func.func @tensor_and_vector_barrier(%write: vector<8xf16>, %input: tensor<8xf16>) -> (vector<8xf16>, vector<8xf16>) {
  %c0 = arith.constant 0 : index
  %cv0 = arith.constant 0.0 : f16

  %wait1 = vector.transfer_write %write, %input[%c0] : vector<8xf16>, tensor<8xf16>
  %synced1 = iree_gpu.value_barrier %wait1 : tensor<8xf16>
  %out = vector.transfer_read %synced1[%c0], %cv0 : tensor<8xf16>, vector<8xf16>

  %synced2 = iree_gpu.value_barrier %write : vector<8xf16>

  return %out, %synced2 : vector<8xf16>, vector<8xf16>
}

// tensor and vector barriers cannot be combined, so both should remain

// CHECK-LABEL: func.func @tensor_and_vector_barrier
// CHECK-COUNT-2: value_barrier
// CHECK-NOT: value_barrier

// -----

func.func @barriers_with_users(%write: vector<8xf16>, %input: tensor<8xf16>, %input2 : tensor<16xf16>, %input3 : tensor<16xf16>) -> (vector<8xf16>) {
  %c0 = arith.constant 0 : index
  %cv0 = arith.constant 0.0 : f16

  %wait1 = vector.transfer_write %write, %input[%c0] : vector<8xf16>, tensor<8xf16>
  %synced1 = iree_gpu.value_barrier %wait1 : tensor<8xf16>
  %out = vector.transfer_read %synced1[%c0], %cv0 : tensor<8xf16>, vector<8xf16>

  %wait2 = vector.transfer_write %write, %input2[%c0] : vector<8xf16>, tensor<16xf16>
  %synced2 = iree_gpu.value_barrier %wait2 : tensor<16xf16>
  %out2 = vector.transfer_read %synced2[%c0], %cv0 : tensor<16xf16>, vector<8xf16>

  %add1 = arith.addf %out, %out2 : vector<8xf16>

  %wait3 = vector.transfer_write %write, %input3[%c0] : vector<8xf16>, tensor<16xf16>
  %synced3 = iree_gpu.value_barrier %wait3 : tensor<16xf16>
  %out3 = vector.transfer_read %synced3[%c0], %cv0 : tensor<16xf16>, vector<8xf16>

  %add2 = arith.addf %add1, %out3 : vector<8xf16>

  return %add2 : vector<8xf16>
}

// There should be only 1 value_barrier left

// CHECK-LABEL: func.func @barriers_with_users
// CHECK: value_barrier
// CHECK-NOT: value_barrier

// -----

func.func @barrier_diamond_chain(%write: vector<8xf16>, %input: tensor<8xf16>) -> (tensor<8xf16>) {
  %c0 = arith.constant 0 : index
  %cv0 = arith.constant 0.0 : f16

  %wait1 = vector.transfer_write %write, %input[%c0] : vector<8xf16>, tensor<8xf16>
  %synced1 = iree_gpu.value_barrier %wait1 : tensor<8xf16>

  %wait2 = vector.transfer_write %write, %synced1[%c0] : vector<8xf16>, tensor<8xf16>
  %synced2 = iree_gpu.value_barrier %wait2 : tensor<8xf16>

  %wait3 = vector.transfer_write %write, %synced1[%c0] : vector<8xf16>, tensor<8xf16>
  %synced3 = iree_gpu.value_barrier %wait3 : tensor<8xf16>

  %d1 = vector.transfer_read %synced2[%c0], %cv0 : tensor<8xf16>, vector<8xf16>
  %d2 = vector.transfer_read %synced3[%c0], %cv0 : tensor<8xf16>, vector<8xf16>

  %add = arith.addf %d1, %d2 : vector<8xf16>

  %synced4 = iree_gpu.value_barrier %add : vector<8xf16>

  %empty = tensor.empty() : tensor<8xf16>
  %out = vector.transfer_write %synced4, %empty[%c0] : vector<8xf16>, tensor<8xf16>

  return %out : tensor<8xf16>
}

// There should be 3 value_barriers left, since in a diamond chain, you can
// only combine the middle barriers.

// CHECK-LABEL: func.func @barrier_diamond_chain
// CHECK-COUNT-3: value_barrier
// CHECK-NOT: value_barrier

// -----

// barrier_region tests

func.func @combine_barrier_region(%arg0: tensor<6xf32>, %arg1: tensor<7xf32>) -> (tensor<1xf32>, tensor<2xf32>) {
  %0 = iree_gpu.barrier_region ins(%arg0 : tensor<6xf32>) {
  ^bb0(%intermediate: tensor<6xf32>):
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<6xf32> to tensor<1xf32>
    iree_gpu.yield %slice : tensor<1xf32>
  } : tensor<1xf32>
  %1 = iree_gpu.barrier_region ins(%arg1 : tensor<7xf32>) {
  ^bb0(%intermediate: tensor<7xf32>):
    %slice = tensor.extract_slice %intermediate[2] [2] [2] : tensor<7xf32> to tensor<2xf32>
    iree_gpu.yield %slice : tensor<2xf32>
  } : tensor<2xf32>
  return %0, %1 : tensor<1xf32>, tensor<2xf32>
}

// CHECK-LABEL: func @combine_barrier_region
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<6xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<7xf32>
//       CHECK:   %[[B:.+]]:2 = iree_gpu.barrier_region ins(%[[ARG0]], %[[ARG1]] : tensor<6xf32>, tensor<7xf32>) {
//       CHECK:     ^bb0(%[[I0:.+]]: tensor<6xf32>, %[[I1:.+]]: tensor<7xf32>):
//       CHECK:       %[[S0:.+]] = tensor.extract_slice %[[I0]][1] [1] [1]
//       CHECK:       %[[S1:.+]] = tensor.extract_slice %[[I1]][2] [2] [2]
//       CHECK:       iree_gpu.yield %[[S0]], %[[S1]] : tensor<1xf32>, tensor<2xf32>
//       CHECK:   } : tensor<1xf32>, tensor<2xf32>
//       CHECK:   return %[[B]]#0, %[[B]]#1

// -----

func.func @combine_three_barrier_regions(
    %arg0: tensor<6xf32>,
    %arg1: tensor<7xf32>,
    %arg2: tensor<8xf32>) -> (tensor<1xf32>, tensor<2xf32>, tensor<3xf32>) {
  %0 = iree_gpu.barrier_region ins(%arg0 : tensor<6xf32>) {
  ^bb0(%intermediate: tensor<6xf32>):
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<6xf32> to tensor<1xf32>
    iree_gpu.yield %slice : tensor<1xf32>
  } : tensor<1xf32>
  %1 = iree_gpu.barrier_region ins(%arg1 : tensor<7xf32>) {
  ^bb0(%intermediate: tensor<7xf32>):
    %slice = tensor.extract_slice %intermediate[2] [2] [2] : tensor<7xf32> to tensor<2xf32>
    iree_gpu.yield %slice : tensor<2xf32>
  } : tensor<2xf32>
  %2 = iree_gpu.barrier_region ins(%arg2 : tensor<8xf32>) {
  ^bb0(%intermediate: tensor<8xf32>):
    %slice = tensor.extract_slice %intermediate[3] [3] [1] : tensor<8xf32> to tensor<3xf32>
    iree_gpu.yield %slice : tensor<3xf32>
  } : tensor<3xf32>
  return %0, %1, %2 : tensor<1xf32>, tensor<2xf32>, tensor<3xf32>
}

// CHECK-LABEL: func @combine_three_barrier_regions
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9]+]]: tensor<6xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: tensor<7xf32>
//  CHECK-SAME:   %[[ARG2:[A-Za-z0-9]+]]: tensor<8xf32>
//       CHECK:   %[[B:.+]]:3 = iree_gpu.barrier_region ins(%[[ARG0]], %[[ARG1]], %[[ARG2]]
//       CHECK:     ^bb0(%[[I0:.+]]: tensor<6xf32>, %[[I1:.+]]: tensor<7xf32>, %[[I2:.+]]: tensor<8xf32>):
//       CHECK:       %[[S0:.+]] = tensor.extract_slice %[[I0]][1] [1] [1]
//       CHECK:       %[[S1:.+]] = tensor.extract_slice %[[I1]][2] [2] [2]
//       CHECK:       %[[S2:.+]] = tensor.extract_slice %[[I2]][3] [3] [1]
//       CHECK:       iree_gpu.yield %[[S0]], %[[S1]], %[[S2]] : tensor<1xf32>, tensor<2xf32>, tensor<3xf32>
//       CHECK:   } : tensor<1xf32>, tensor<2xf32>, tensor<3xf32>
//       CHECK:   return %[[B]]#0, %[[B]]#1, %[[B]]#2

// -----

func.func @dont_combine_dependent_barriers(%arg0: tensor<6xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
  %0 = iree_gpu.barrier_region ins(%arg0 : tensor<6xf32>) {
  ^bb0(%intermediate: tensor<6xf32>):
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<6xf32> to tensor<1xf32>
    iree_gpu.yield %slice : tensor<1xf32>
  } : tensor<1xf32>
  %1 = iree_gpu.barrier_region ins(%0 : tensor<1xf32>) {
  ^bb0(%intermediate: tensor<1xf32>):
    iree_gpu.yield %intermediate : tensor<1xf32>
  } : tensor<1xf32>
  return %0, %1 : tensor<1xf32>, tensor<1xf32>
}

// CHECK-LABEL: func @dont_combine_dependent_barriers
// CHECK-COUNT-2:   iree_gpu.barrier_region

// -----

func.func @dont_combine_implicit_capture(%arg0: tensor<6xf32>, %arg1: tensor<7xf32>) -> (tensor<1xf32>, tensor<2xf32>) {
  %0 = iree_gpu.barrier_region ins(%arg0 : tensor<6xf32>) {
  ^bb0(%intermediate: tensor<6xf32>):
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<6xf32> to tensor<1xf32>
    iree_gpu.yield %slice : tensor<1xf32>
  } : tensor<1xf32>
  %1 = iree_gpu.barrier_region ins(%arg1 : tensor<7xf32>) {
  ^bb0(%intermediate: tensor<7xf32>):
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<7xf32> to tensor<1xf32>
    %concat = tensor.concat dim(0) %slice, %0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    iree_gpu.yield %concat : tensor<2xf32>
  } : tensor<2xf32>
  return %0, %1 : tensor<1xf32>, tensor<2xf32>
}

// CHECK-LABEL: func @dont_combine_implicit_capture
// CHECK-COUNT-2:   iree_gpu.barrier_region

// -----

// The key case: non-adjacent barrier_regions with an intervening op that
// doesn't depend on either barrier. The old CombineBarrierRegions pass
// could not handle this.

func.func @combine_nonadjacent_barrier_regions(
    %arg0: tensor<6xf32>, %arg1: tensor<7xf32>) -> (tensor<1xf32>, tensor<2xf32>) {
  %c0 = arith.constant 0 : index
  %0 = iree_gpu.barrier_region ins(%arg0 : tensor<6xf32>) {
  ^bb0(%intermediate: tensor<6xf32>):
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<6xf32> to tensor<1xf32>
    iree_gpu.yield %slice : tensor<1xf32>
  } : tensor<1xf32>
  // Intervening op that does not depend on either barrier.
  %dummy = arith.addi %c0, %c0 : index
  %1 = iree_gpu.barrier_region ins(%arg1 : tensor<7xf32>) {
  ^bb0(%intermediate: tensor<7xf32>):
    %slice = tensor.extract_slice %intermediate[2] [2] [2] : tensor<7xf32> to tensor<2xf32>
    iree_gpu.yield %slice : tensor<2xf32>
  } : tensor<2xf32>
  return %0, %1 : tensor<1xf32>, tensor<2xf32>
}

// CHECK-LABEL: func @combine_nonadjacent_barrier_regions
// CHECK: iree_gpu.barrier_region
// CHECK-NOT: iree_gpu.barrier_region
