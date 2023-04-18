// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @add_2d_f32
func.func @add_2d_f32(
    // LHS
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index, %arg3 : index,
    // RHS
    %arg4 : !util.buffer, %arg5 : index, %arg6 : index, %arg7 : index,
    // OUT
    %arg8 : !util.buffer, %arg9 : index, %arg10 : index, %arg11 : index,
    // SIZE
    %arg12 : index, %arg13 : index) {

  //      CHECK: vm.call @vmvx.add.2d.f32(
  // CHECK-SAME:   %arg0, %arg1, %arg2, %arg3,
  // CHECK-SAME:   %arg4, %arg5, %arg6, %arg7,
  // CHECK-SAME:   %arg8, %arg9, %arg10, %arg11,
  // CHECK-SAME:   %arg12, %arg13)
  // CHECK-SAME: : (!vm.buffer, i64, i64, i64, !vm.buffer, i64, i64, i64, !vm.buffer, i64, i64, i64, i64, i64) -> ()
  vmvx.binary op("add" : f32)
           lhs(%arg0 offset %arg1 strides[%arg2, %arg3] : !util.buffer)
           rhs(%arg4 offset %arg5 strides[%arg6, %arg7] : !util.buffer)
           out(%arg8 offset %arg9 strides[%arg10, %arg11] : !util.buffer)
           sizes(%arg12, %arg13)
  func.return
}
