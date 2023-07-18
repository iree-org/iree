// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @fill2d_f32
func.func @fill2d_f32(%arg0 : !util.buffer, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : f32) {
  // CHECK-DAG: %[[VALUE:.*]] = vm.bitcast.f32.i32 %arg5 : f32 -> i32
  //     CHECK: vm.call @vmvx.fill.2d.x32(%[[VALUE]], %arg0, %arg1, %arg2, %arg3, %arg4) : (i32, !vm.buffer, i64, i64, i64, i64) -> ()
  vmvx.fill2d scalar(%arg5 : f32) out(%arg0 offset %arg1 row_stride %arg2 : !util.buffer) sizes(%arg3, %arg4)
  func.return
}

// CHECK-LABEL: @fill2d_i32
func.func @fill2d_i32(%arg0 : !util.buffer, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index, %arg5 : i32) {
  //     CHECK: vm.call @vmvx.fill.2d.x32(%arg5, {{.*}}) -> ()
  vmvx.fill2d scalar(%arg5 : i32) out(%arg0 offset %arg1 row_stride %arg2 : !util.buffer) sizes(%arg3, %arg4)
  func.return
}
