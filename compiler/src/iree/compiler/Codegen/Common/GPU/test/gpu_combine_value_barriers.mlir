// RUN: iree-opt --iree-codegen-gpu-combine-value-barriers %s --split-input-file | FileCheck %s

// Since the pass only rearanges the order of instructions, we only check the
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
// CHECK: value_barrier
// CHECK: value_barrier

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
// CHECK: value_barrier
// CHECK: value_barrier
// CHECK: value_barrier
// CHECK-NOT: value_barrier
