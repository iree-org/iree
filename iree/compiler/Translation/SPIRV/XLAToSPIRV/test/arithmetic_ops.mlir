// RUN: iree-opt -split-input-file -iree-index-computation -simplify-spirv-affine-exprs=false -convert-iree-to-spirv -verify-diagnostics -o - %s | IreeFileCheck %s

// CHECK-DAG: spv.globalVariable [[GLOBALIDVAR:@.*]] built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
// CHECK: spv.func @mul_1D
// CHECK-SAME: [[ARG0:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
// CHECK-SAME: [[ARG1:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
// CHECK-SAME: [[ARG2:%[a-zA-Z0-9]*]]: !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
func @mul_1D(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
  // CHECK: [[ZERO1:%.*]] = spv.constant 0 : i32
  // CHECK: [[ARG0LOADPTR:%.*]] = spv.AccessChain [[ARG0]]{{\[}}[[ZERO1]], {{.*}}{{\]}}
  // CHECK: [[VAL1:%.*]] = spv.Load "StorageBuffer" [[ARG0LOADPTR]]
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  // CHECK: [[ZERO2:%.*]] = spv.constant 0 : i32
  // CHECK: [[ARG1LOADPTR:%.*]] = spv.AccessChain [[ARG1]]{{\[}}[[ZERO2]], {{.*}}{{\]}}
  // CHECK: [[VAL2:%.*]] = spv.Load "StorageBuffer" [[ARG1LOADPTR]]
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: [[RESULT:%.*]] = spv.FMul [[VAL1]], [[VAL2]]
  %2 = mulf %0, %1 : tensor<4xf32>
  // CHECK: [[ZERO3:%.*]] = spv.constant 0 : i32
  // CHECK: [[ARG2STOREPTR:%.*]] = spv.AccessChain [[ARG2]]{{\[}}[[ZERO3]], {{.*}}{{\]}}
  // CHECK: spv.Store "StorageBuffer" [[ARG2STOREPTR]], [[RESULT]]
  iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
  return
}

// -----

func @frem(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
  %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
  // CHECK: spv.FRem
  %2 = remf %0, %1 : tensor<4xf32>
  iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
  return
}

// -----

func @srem(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>)
attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.SRem
  %2 = remi_signed %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi32>, %arg2 : memref<4xi32>)
  return
}

// -----

func @srem(%arg0: memref<4xi32>, %arg1: memref<4xi32>, %arg2: memref<4xi32>)
attributes  {iree.executable.export, iree.executable.workgroup_size = [1 : index, 1 : index, 1 : index], iree.ordinal = 0 : i32} {
  %0 = iree.load_input(%arg0 : memref<4xi32>) : tensor<4xi32>
  %1 = iree.load_input(%arg1 : memref<4xi32>) : tensor<4xi32>
  // CHECK: spv.SRem
  %2 = remi_unsigned %0, %1 : tensor<4xi32>
  iree.store_output(%2 : tensor<4xi32>, %arg2 : memref<4xi32>)
  return
}
