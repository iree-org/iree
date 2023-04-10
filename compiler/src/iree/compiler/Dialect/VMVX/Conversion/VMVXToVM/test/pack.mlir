// RUN: iree-opt --iree-vm-target-index-bits=64 --split-input-file \
// RUN:   --iree-vm-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @pack_f32f32
func.func @pack_f32f32(
    // IN buffer/offset/stride0
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // OUT buffer/offset/stride0
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // IN size0/size1
    %arg6 : index, %arg7 : index,
    // OUT size0/size1/size2/size3
    %arg8 : index, %arg9 : index, %arg10 : index, %arg11 : index,
    // padding value
    %arg12 : f32
    ) {
  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32.zero
  //      CHECK: vm.call @vmvx.pack.f32f32(
  // CHECK-SAME: %arg0, %arg1, %arg3, %arg4,
  // CHECK-SAME: %arg2, %arg5,
  // CHECK-SAME: %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %arg12,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, i32) -> ()
  vmvx.pack in(%arg0 offset %arg1 stride0 %arg2 : !util.buffer)
             out(%arg3 offset %arg4 stride0 %arg5 : !util.buffer)
             in_shape(%arg6, %arg7)
             out_shape(%arg8, %arg9, %arg10, %arg11)
             padding_value(%arg12 : f32)
             flags(0) : (f32, f32)
  func.return
}


// CHECK-LABEL: @pack_i32i32
func.func @pack_i32i32(
    // IN buffer/offset/stride0
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // OUT buffer/offset/stride0
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // IN size0/size1
    %arg6 : index, %arg7 : index,
    // OUT size0/size1/size2/size3
    %arg8 : index, %arg9 : index, %arg10 : index, %arg11 : index,
    // padding value
    %arg12 : i32
    ) {
  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32 65536
  //      CHECK: vm.call @vmvx.pack.i32i32(
  // CHECK-SAME: %arg0, %arg1, %arg3, %arg4,
  // CHECK-SAME: %arg2, %arg5,
  // CHECK-SAME: %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %arg12,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32) -> ()
  vmvx.pack in(%arg0 offset %arg1 stride0 %arg2 : !util.buffer)
             out(%arg3 offset %arg4 stride0 %arg5 : !util.buffer)
             in_shape(%arg6, %arg7)
             out_shape(%arg8, %arg9, %arg10, %arg11)
             padding_value(%arg12 : i32)
             flags(65536) : (i32, i32)
  func.return
}

// CHECK-LABEL: @pack_i8i8
func.func @pack_i8i8(
    // IN buffer/offset/stride0
    %arg0 : !util.buffer, %arg1 : index, %arg2 : index,
    // OUT buffer/offset/stride0
    %arg3 : !util.buffer, %arg4 : index, %arg5 : index,
    // IN size0/size1
    %arg6 : index, %arg7 : index,
    // OUT size0/size1/size2/size3
    %arg8 : index, %arg9 : index, %arg10 : index, %arg11 : index,
    // padding value
    %arg12 : i32
    ) {
  //  CHECK-DAG: %[[FLAGS:.*]] = vm.const.i32.zero
  //      CHECK: vm.call @vmvx.pack.i8i8(
  // CHECK-SAME: %arg0, %arg1, %arg3, %arg4,
  // CHECK-SAME: %arg2, %arg5,
  // CHECK-SAME: %arg6, %arg7, %arg8,
  // CHECK-SAME: %arg9, %arg10, %arg11,
  // CHECK-SAME: %arg12,
  // CHECK-SAME: %[[FLAGS]]) : (!vm.buffer, i64, !vm.buffer, i64, i64, i64, i64, i64, i64, i64, i64, i64, i32, i32) -> ()
  vmvx.pack in(%arg0 offset %arg1 stride0 %arg2 : !util.buffer)
             out(%arg3 offset %arg4 stride0 %arg5 : !util.buffer)
             in_shape(%arg6, %arg7)
             out_shape(%arg8, %arg9, %arg10, %arg11)
             padding_value(%arg12 : i32)
             flags(0) : (i8, i8)
  func.return
}
