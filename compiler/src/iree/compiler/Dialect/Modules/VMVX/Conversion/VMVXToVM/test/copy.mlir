// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @copy_2d_f32
func.func @copy_2d_f32(
    // INP
    %arg0 : memref<65536xf32>, %arg1 : index, %arg2 : index, %arg3 : index,
    // OUT
    %arg4 : memref<65536xf32>, %arg5 : index, %arg6 : index, %arg7 : index,
    // SIZE
    %arg8 : index, %arg9 : index) {

  //      CHECK: vm.call @vmvx.copy.2d.x32(
  // CHECK-SAME:   %arg0, %arg1, %arg2, %arg3,
  // CHECK-SAME:   %arg4, %arg5, %arg6, %arg7,
  // CHECK-SAME:   %arg8, %arg9)
  // CHECK-SAME: : (!vm.buffer, i64, i64, i64, !vm.buffer, i64, i64, i64, i64, i64) -> ()
  vmvx.copy in(%arg0 offset %arg1 strides[%arg2, %arg3] : memref<65536xf32>)
            out(%arg4 offset %arg5 strides[%arg6, %arg7] : memref<65536xf32>)
            sizes(%arg8, %arg9)
  func.return
}
