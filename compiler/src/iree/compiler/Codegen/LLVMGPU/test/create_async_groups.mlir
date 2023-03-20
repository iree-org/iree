// RUN: iree-opt %s -iree-transform-dialect-interpreter -split-input-file --verify-diagnostics | FileCheck %s

// Check that we produce async copies from the vector.transfer_xxx operations.
builtin.module {
  // CHECK-LABEL: @copies_to_asyncs
  func.func @copies_to_asyncs(%a: memref<1024x1024xf32>) {
    %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    // Make sure we emit the bypassL1.
    // CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4  {bypassL1} :
    %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
    vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK-NOT: nvgpu.device_async_create_group

    // CHECK: %[[CP1:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 1  {bypassL1} :
    %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
    vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]], %[[CP1]]
    // CHECK: nvgpu.device_async_wait %[[G]]
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !pdl.operation):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    transform.iree.create_async_groups %top_level_func {use_mma_sync = true} : (!pdl.operation) -> ()
  }
}

// -----

// Check that we properly take `use_mma_sync = false` into account.
// I.e., we shouldn't be generating bypassL1 attributes.
builtin.module {
  // CHECK-LABEL: @copies_to_asyncs_no_mma
  func.func @copies_to_asyncs_no_mma(%a: memref<1024x1024xf32>) {
    %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    // Make sure we don't emit the bypassL1.
    // CHECK: %[[CP0:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 4 :
    %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
    vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK-NOT: nvgpu.device_async_create_group

    // CHECK: %[[CP1:.*]] = nvgpu.device_async_copy {{.*}}, {{.*}}, 1 :
    %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
    vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    // CHECK: %[[G:.*]] = nvgpu.device_async_create_group %[[CP0]], %[[CP1]]
    // CHECK: nvgpu.device_async_wait %[[G]]
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !pdl.operation):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    transform.iree.create_async_groups %top_level_func {use_mma_sync = false} : (!pdl.operation) -> ()
  }
}

// -----

// Check that we reject constructs that try to apply create_async_groups
// on non-func op.

// expected-error@below {{transform dialect interpreter failed}}
builtin.module {
  func.func @copies_to_asyncs_invalid_op_input(%a: memref<1024x1024xf32>) {
    // expected-note@below {{when applied to this op}}
    %0 = memref.alloc() : memref<4x32x16xf32, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %1 = vector.transfer_read %a[%c0, %c0], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<4xf32>
    vector.transfer_write %1, %0[%c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>

    %2 = vector.transfer_read %a[%c0, %c4], %cst_0 {in_bounds = [true]} : memref<1024x1024xf32>, vector<1xf32>
    vector.transfer_write %2, %0[%c0, %c4, %c0] {in_bounds = [true]} : vector<1xf32>, memref<4x32x16xf32, #gpu.address_space<workgroup>>
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !pdl.operation):
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %vector_transfer = transform.structured.match ops{["memref.alloc"]} in %top_level_func : (!pdl.operation) -> !pdl.operation
    // expected-error@below {{transform applied to the wrong op kind}}
    transform.iree.create_async_groups %vector_transfer {use_mma_sync = false} : (!pdl.operation) -> ()
  }
}

