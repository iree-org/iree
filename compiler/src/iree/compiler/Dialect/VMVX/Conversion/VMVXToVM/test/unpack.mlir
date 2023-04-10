// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @unpack_f32f32
func.func @unpack_f32f32(
    // IN buffer/offset/stride0
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // OUT buffer/offset/stride0
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // IN size0/size1/size2/size3
    %arg6 : index, %arg7 : index, %arg8 : index, %arg9 : index,
    // OUT size0/size1
    %arg10 : index, %arg11 : index
    ) {
  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32.zero
  //      CHECK: vm.call @vmvx.unpack.f32f32(
  // CHECK-SAME: %arg0, %arg1, %arg3,  %arg4, %arg2,
  // CHECK-SAME: %arg5, %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> ()
  vmvx.unpack in(%arg0 offset %arg1 stride0 %arg2 : !util.buffer)
             out(%arg3 offset %arg4 stride0 %arg5 : !util.buffer)
             in_shape(%arg6, %arg7, %arg8, %arg9)
             out_shape(%arg10, %arg11)
             flags(0) : (f32, f32)
  func.return
}


// CHECK-LABEL: @unpack_i32i32
func.func @unpack_i32i32(
    // IN buffer/offset/stride0
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // OUT buffer/offset/stride0
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // IN size0/size1/size2/size3
    %arg6 : index, %arg7 : index, %arg8 : index, %arg9 : index,
    // OUT size0/size1
    %arg10 : index, %arg11 : index
    ) {
  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32 65536
  //      CHECK: vm.call @vmvx.unpack.i32i32(
  // CHECK-SAME: %arg0, %arg1, %arg3,  %arg4, %arg2,
  // CHECK-SAME: %arg5, %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> ()
  vmvx.unpack in(%arg0 offset %arg1 stride0 %arg2 : !util.buffer)
             out(%arg3 offset %arg4 stride0 %arg5 : !util.buffer)
             in_shape(%arg6, %arg7, %arg8, %arg9)
             out_shape(%arg10, %arg11)
             flags(65536) : (i32, i32)
  func.return
}

// CHECK-LABEL: @unpack_i8i8
func.func @unpack_i8i8(
    // IN buffer/offset/stride0
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // OUT buffer/offset/stride0
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // IN size0/size1/size2/size3
    %arg6 : index, %arg7 : index,  %arg8 : index, %arg9 : index,
    // OUT size0/size1
    %arg10 : index, %arg11 : index
    ) {
  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32.zero
  //      CHECK: vm.call @vmvx.unpack.i8i8(
  // CHECK-SAME: %arg0, %arg1, %arg3,  %arg4, %arg2,
  // CHECK-SAME: %arg5, %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32) -> ()
  vmvx.unpack in(%arg0 offset %arg1 stride0 %arg2 : !util.buffer)
             out(%arg3 offset %arg4 stride0 %arg5 : !util.buffer)
             in_shape(%arg6, %arg7, %arg8, %arg9)
             out_shape(%arg10, %arg11)
             flags(0) : (i8, i8)
  func.return
}
