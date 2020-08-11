// RUN: iree-opt -split-input-file -iree-convert-vmla-to-vm -cse %s | IreeFileCheck %s

// CHECK: vm.rodata @[[CONST_SYM:.+]] dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
// CHECK-NEXT: vm.func @constValues
func @constValues() -> !vmla.buffer {
  // CHECK-NEXT: %[[BYTES_REF:.+]] = vm.const.ref.rodata @[[CONST_SYM]] : !vm.ref<!iree.byte_buffer>
  // CHECK-NEXT: %[[BUFFER:.+]] = vm.call @vmla.buffer.const(%[[BYTES_REF]]) : (!vm.ref<!iree.byte_buffer>) -> !vm.ref<!vmla.buffer>
  %0 = vmla.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32> -> !vmla.buffer
  // CHECK-NEXT: vm.return %[[BUFFER]] : !vm.ref<!vmla.buffer>
  return %0 : !vmla.buffer
}

// -----

// CHECK-LABEL: vm.func @bufferImport
func @bufferImport() -> !vmla.buffer {
  %c0 = std.constant 1 : index
  // CHECK: = vm.call @vmla.buffer.alloc(%c1) : (i32) -> !vm.ref<!vmla.buffer>
  %0 = vmla.buffer.alloc byte_length = %c0 : !vmla.buffer
  return %0 : !vmla.buffer
}

// -----

// CHECK-LABEL: vm.func @typedImport
func @typedImport(%arg0 : !vmla.buffer, %arg1 : !vmla.buffer) {
  // CHECK-NEXT: %c1 = vm.const.i32 1 : i32
  // CHECK-NEXT: vm.call @vmla.cmp.f32(%c1, %arg0, %arg0, %arg1) : (i32, !vm.ref<!vmla.buffer>, !vm.ref<!vmla.buffer>, !vm.ref<!vmla.buffer>) -> ()
  vmla.cmp "NE", %arg0, %arg0, out %arg1 : f32
  return
}

// -----

// CHECK-LABEL: vm.func @sizedImport
func @sizedImport(%arg0 : !vmla.buffer, %arg1 : !vmla.buffer) {
  // CHECK-NEXT: vm.call @vmla.select.x32(%arg0, %arg0, %arg0, %arg1)
  vmla.select %arg0, %arg0, %arg0, out %arg1 : f32
  return
}

// -----

// CHECK-LABEL: vm.func @shapeExpansion
// CHECK-SAME: %arg0: !vm.ref<!vmla.buffer>, %arg1: i32, %arg2: !vm.ref<!vmla.buffer>, %arg3: i32
func @shapeExpansion(%arg0 : !vmla.buffer, %arg1 : index, %arg2 : !vmla.buffer, %arg3 : index) {
  %rs0 = shapex.make_ranked_shape %arg1 : (index) -> !shapex.ranked_shape<[4,?,8]>
  %rs1 = shapex.make_ranked_shape %arg3 : (index) -> !shapex.ranked_shape<[?,4,8]>
  // CHECK-DAG: %c1 = vm.const.i32 1 : i32
  // CHECK-DAG: %c4 = vm.const.i32 4 : i32
  // CHECK-DAG: %c8 = vm.const.i32 8 : i32
  // CHECK-NEXT: vm.call.variadic @vmla.transpose.x16(%arg0, [%c4, %arg1, %c8], [%c1], %arg2, [%arg3, %c4, %c8]) : (!vm.ref<!vmla.buffer>, i32 ..., i32 ..., !vm.ref<!vmla.buffer>, i32 ...)
  vmla.transpose %arg0(%rs0 : !shapex.ranked_shape<[4,?,8]>),
                 out %arg2(%rs1 : !shapex.ranked_shape<[?,4,8]>)
                 {permutation = dense<[1]> : tensor<1xi32>} : i16
  return
}

// -----

// CHECK-LABEL: vm.func @convert
func @convert(%arg0 : !vmla.buffer, %arg1 : !vmla.buffer) {
  // CHECK-NEXT:  vm.call @vmla.convert.f32.i8(%arg0, %arg1)
  vmla.convert %arg0, out %arg1 : f32 -> i8
  return
}

// -----

// CHECK-LABEL: vm.func @batch_matmul
func @batch_matmul(
    %lhs : !vmla.buffer,
    %lhs_dim : index,
    %rhs : !vmla.buffer,
    %rhs_dim : index,
    %dst : !vmla.buffer) {
  %lhs_shape = shapex.make_ranked_shape %lhs_dim : (index) -> !shapex.ranked_shape<[3,4,?]>
  %rhs_shape = shapex.make_ranked_shape %rhs_dim : (index) -> !shapex.ranked_shape<[3,?,4]>
  %dst_shape = shapex.const_ranked_shape : !shapex.ranked_shape<[3,4,4]>
  // CHECK: vm.call.variadic @vmla.batch.matmul.f32f32.f32(%arg0, [%c3, %c4, %arg1], %arg2, [%c3, %arg3, %c4], %arg4, [%c3, %c4, %c4])
  vmla.batch.matmul %lhs(%lhs_shape : !shapex.ranked_shape<[3,4,?]>) : f32,
                    %rhs(%rhs_shape : !shapex.ranked_shape<[3,?,4]>) : f32,
                    out %dst(%dst_shape : !shapex.ranked_shape<[3,4,4]>) : f32
  return
}
