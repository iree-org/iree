// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @matmul_f32f32f32
func.func @matmul_f32f32f32(
    // LHS
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // RHS
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // OUT
    %arg6 : !util.buffer, %arg7 : index, %arg8 : index,
    // SIZE
    %arg9 : index, %arg10 : index, %arg11 : index,
    // SCALE
    %arg12 : f32, %arg13 : f32) {

  //  CHECK-DAG: %[[ZERO:.*]] = vm.const.i32.zero
  //      CHECK: vm.call @vmvx.matmul.f32f32f32(
  // CHECK-SAME: %arg0, %arg1, %arg2,
  // CHECK-SAME: %arg3, %arg4, %arg5,
  // CHECK-SAME: %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11, %arg12, %arg13, %[[ZERO]]) : (!vm.buffer, i64, i64, !vm.buffer, i64, i64, !vm.buffer, i64, i64, i64, i64, i64, f32, f32, i32) -> ()
  vmvx.matmul lhs(%arg0 offset %arg1 row_stride %arg2 : !util.buffer)
              rhs(%arg3 offset %arg4 row_stride %arg5 : !util.buffer)
              out(%arg6 offset %arg7 row_stride %arg8 : !util.buffer)
              mnk(%arg9, %arg10, %arg11)
              scale(%arg12 : f32, %arg13 : f32)
              flags(0) : (f32, f32, f32)
  func.return
}
