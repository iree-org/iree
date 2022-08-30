// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @mmt4d_f32f32f32
func.func @mmt4d_f32f32f32(
    // LHS
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // RHS
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // OUT
    %arg6 : !util.buffer, %arg7 : index, %arg8 : index,
    // MNK
    %arg9 : index, %arg10 : index, %arg11 : index,
    // TILE_MNK
    %arg12 : index, %arg13 : index, %arg14 : index
    ) {

  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32 1
  //      CHECK: vm.call @vmvx.mmt4d.f32f32f32(
  // CHECK-SAME: %arg0, %arg1, %arg2,
  // CHECK-SAME: %arg3, %arg4, %arg5,
  // CHECK-SAME: %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, i64, !vm.buffer, i64, i64, !vm.buffer, i64, i64, i64, i64, i64, i32, i32, i32, i32) -> ()
  vmvx.mmt4d lhs(%arg0 offset %arg1 row_stride %arg2 : !util.buffer)
             rhs(%arg3 offset %arg4 row_stride %arg5 : !util.buffer)
             out(%arg6 offset %arg7 row_stride %arg8 : !util.buffer)
             mnk(%arg9, %arg10, %arg11)
             tile_mnk(%arg12, %arg13, %arg14)
             flags(1) : (f32, f32, f32)
  func.return
}

// CHECK-LABEL: @mmt4d_i8i8i32
func.func @mmt4d_i8i8i32(
    // LHS
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // RHS
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // OUT
    %arg6 : !util.buffer, %arg7 : index, %arg8 : index,
    // MNK
    %arg9 : index, %arg10 : index, %arg11 : index,
    // TILE_MNK
    %arg12 : index, %arg13 : index, %arg14 : index
    ) {

  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32 1
  //      CHECK: vm.call @vmvx.mmt4d.i8i8i32(
  // CHECK-SAME: %arg0, %arg1, %arg2,
  // CHECK-SAME: %arg3, %arg4, %arg5,
  // CHECK-SAME: %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, i64, !vm.buffer, i64, i64, !vm.buffer, i64, i64, i64, i64, i64, i32, i32, i32, i32) -> ()
  vmvx.mmt4d lhs(%arg0 offset %arg1 row_stride %arg2 : !util.buffer)
             rhs(%arg3 offset %arg4 row_stride %arg5 : !util.buffer)
             out(%arg6 offset %arg7 row_stride %arg8 : !util.buffer)
             mnk(%arg9, %arg10, %arg11)
             tile_mnk(%arg12, %arg13, %arg14)
             flags(1) : (i8, i8, i32)
  func.return
}
