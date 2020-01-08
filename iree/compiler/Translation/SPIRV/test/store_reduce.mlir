// RUN: iree-opt -split-input-file -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

#map0 = (d0, d1, d2) -> (d0)
module {
  // CHECK: func @reduction_entry
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<5 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  func @reduction_entry(%arg0: memref<5xi32> {iree.index_computation_info = [[#map0]]}, %arg1: memref<i32>, %arg2: memref<i32> {iree.executable.reduction.output, iree.index_computation_info = [[(d0, d1, d2) -> (0)]]}) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 0 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[5, 1, 1]> : tensor<3xi32>, iree.num_dims = 3 : i32, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<5xi32>)  {iree.index_computation_info = [[#map0, #map0]]} : tensor<5xi32>
    // CHECK: [[LOADPTR:%[a-zA-Z0-9_]*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[LOADVAL:%[a-zA-Z0-9_]*]]  = spv.Load "StorageBuffer" [[LOADPTR]]
    // CHECK: [[STOREPTR:%[a-zA-Z0-9_]*]] = spv.AccessChain [[ARG2]]
    // CHECK: spv.FunctionCall @reduction_apply([[LOADVAL]], [[STOREPTR]])
    iree.store_reduce(%0 : tensor<5xi32>, %arg2 : memref<i32>, @reduction_apply)
    iree.return
  }
  func @reduction_apply(%arg0: i32, %arg1: !spv.ptr<i32, StorageBuffer>) {
    %0 = spv.AtomicSMax "Device" "None" %arg1, %arg0 : !spv.ptr<i32, StorageBuffer>
    spv.Return
  }
}

// -----

#map0 = (d0, d1) -> (d1, d0)
module {
  // CHECK: func @reduction_2D_dim0_entry
  // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<20 x i32 [4]> [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<i32 [0]>, StorageBuffer>
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<4 x i32 [4]> [0]>, StorageBuffer>
  func @reduction_2D_dim0_entry(%arg0: memref<5x4xi32> {iree.index_computation_info = [[#map0]]}, %arg1: memref<i32>, %arg2: memref<4xi32> {iree.executable.reduction.output, iree.index_computation_info = [[(d0, d1) -> (d0)]]}) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 0 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[4, 5, 1]> : tensor<3xi32>, iree.num_dims = 2 : i32, iree.ordinal = 0 : i32} {
    %0 = iree.load_input(%arg0 : memref<5x4xi32>)  {iree.index_computation_info = [[#map0, #map0]]} : tensor<5x4xi32>
    // CHECK: [[LOADPTR:%[a-zA-Z0-9_]*]] = spv.AccessChain [[ARG0]]
    // CHECK: [[LOADVAL:%[a-zA-Z0-9_]*]]  = spv.Load "StorageBuffer" [[LOADPTR]]
    // CHECK: [[STOREPTR:%[a-zA-Z0-9_]*]] = spv.AccessChain [[ARG2]]
    // CHECK: spv.FunctionCall @reduction_apply([[LOADVAL]], [[STOREPTR]])
    iree.store_reduce(%0 : tensor<5x4xi32>, %arg2 : memref<4xi32>, @reduction_apply)
    iree.return
  }
  func @reduction_apply(%arg0: i32, %arg1: !spv.ptr<i32, StorageBuffer>) {
    %0 = spv.AtomicSMax "Device" "None" %arg1, %arg0 : !spv.ptr<i32, StorageBuffer>
    spv.Return
  }
}

// -----

#map0 = (d0, d1) -> (d1, d0)
module {
  // CHECK: func @reduction_2D_dim1_entry
  // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]: !spv.ptr<!spv.struct<!spv.array<5 x i32 [4]> [0]>, StorageBuffer>
  func @reduction_2D_dim1_entry(%arg0: memref<5x4xi32> {iree.index_computation_info = [[#map0]]}, %arg1: memref<i32>, %arg2: memref<5xi32> {iree.executable.reduction.output, iree.index_computation_info = [[(d0, d1) -> (d1)]]}) attributes {iree.executable.export, iree.executable.reduction, iree.executable.reduction.apply = @reduction_apply, iree.executable.reduction.dimension = 1 : i32, iree.executable.workgroup_size = dense<[32, 1, 1]> : tensor<3xi64>, iree.executable.workload = dense<[4, 5, 1]> : tensor<3xi32>, iree.num_dims = 2 : i32, iree.ordinal = 0 : i32} {
    // CHECK: [[GLOBALIDPTR:%[a-zA-Z0-9_]*]] = spv._address_of @globalInvocationID
    // CHECK: [[GLOBALID:%[a-zA-Z0-9_]*]] = spv.Load "Input" [[GLOBALIDPTR]] : vector<3xi32>
    // CHECK: [[GLOBALIDY:%[a-zA-Z0-9_]*]] = spv.CompositeExtract [[GLOBALID]]{{\[}}1 : i32{{\]}}
    %0 = iree.load_input(%arg0 : memref<5x4xi32>)  {iree.index_computation_info = [[#map0, #map0]]} : tensor<5x4xi32>
    // CHECK: spv.AccessChain [[ARG2]]{{\[}}{{.*}}, [[GLOBALIDY]]{{\]}}
    iree.store_reduce(%0 : tensor<5x4xi32>, %arg2 : memref<5xi32>, @reduction_apply)
    iree.return
  }
  func @reduction_apply(%arg0: i32, %arg1: !spv.ptr<i32, StorageBuffer>) {
    %0 = spv.AtomicSMax "Device" "None" %arg1, %arg0 : !spv.ptr<i32, StorageBuffer>
    spv.Return
  }
}
