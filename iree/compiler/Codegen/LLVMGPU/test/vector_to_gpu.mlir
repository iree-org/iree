// RUN: iree-opt %s -allow-unregistered-dialect -iree-llvmgpu-vector-to-gpu -canonicalize | FileCheck %s

// CHECK-LABEL: func @copies_to_asyncs
func @copies_to_asyncs(%a: memref<1024x1024xf32>) {
  %0 = memref.alloc() : memref<4x32x16xf32, 3>
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  // CHECK: %[[CP0:.*]] = gpu.device_async_copy {{.*}}, {{.*}}, 4
  %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
  vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, 3>
  // CHECK-NOT: gpu.device_async_create_group

  // CHECK: %[[CP1:.*]] = gpu.device_async_copy {{.*}}, {{.*}}, 1
  %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
  vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, 3>
  // CHECK: %[[G:.*]] = gpu.device_async_create_group %[[CP0]], %[[CP1]]
  // CHECK: gpu.device_async_wait %[[G]]
  return
}
