// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors --canonicalize -cse %s | FileCheck %s

// CHECK-LABEL: @replica_id
func.func @replica_id() -> tensor<ui32> {
  // CHECK-DAG: [[CHANNEL:%.+]] = flow.channel.default : !flow.channel
  // CHECK-DAG: [[RANK:%.+]] = flow.channel.rank [[CHANNEL]] : index
  // CHECK-DAG: [[CAST:%.+]] = arith.index_castui [[RANK]] : index to i32
  // CHECK-DAG: [[TENSOR:%.+]] = tensor.from_elements [[CAST]] : tensor<i32>
  // CHECK-DAG: return [[TENSOR]] : tensor<i32>
  %id = mhlo.replica_id : tensor<ui32>
  return %id : tensor<ui32>
}
