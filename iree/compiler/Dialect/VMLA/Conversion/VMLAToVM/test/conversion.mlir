// RUN: iree-opt -split-input-file -iree-convert-vmla-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: vm.func @bufferImport
func @bufferImport() -> !iree.ref<!vmla.buffer> {
  %c0 = std.constant 1 : i32
  // CHECK: = vm.call @vmla.buffer.alloc(%c1) : (i32) -> !iree.ref<!vmla.buffer>
  %0 = "vmla.buffer.alloc"(%c0) : (i32) -> !iree.ref<!vmla.buffer>
  return %0 : !iree.ref<!vmla.buffer>
}

// -----

// CHECK-LABEL: vm.func @typedImport
func @typedImport(%src : !iree.ref<!vmla.buffer>, %dst : !iree.ref<!vmla.buffer>) {
  // CHECK-NEXT: %c1 = vm.const.i32 1 : i32
  // CHECK-NEXT: vm.call @vmla.cmp.f32(%c1, %arg0, %arg0, %arg1) : (i32, !iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>) -> ()
  "vmla.cmp"(%src, %src, %dst) { predicate = 1 : i32, element_type = f32 } : (!iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>) -> ()
  return
}

// -----

// CHECK-LABEL: vm.func @sizedImport
func @sizedImport(%src : !iree.ref<!vmla.buffer>, %dst : !iree.ref<!vmla.buffer>) {
  // CHECK-NEXT: vm.call @vmla.select.x32(%arg0, %arg0, %arg0, %arg1)
  "vmla.select"(%src, %src, %src, %dst) { element_type = f32 } : (!iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>) -> ()
  return
}

// -----

// CHECK-LABEL: vm.func @shapeExpansion
// CHECK-SAME: %arg0: !iree.ref<!vmla.buffer>, %arg1: i32, %arg2: !iree.ref<!vmla.buffer>
func @shapeExpansion(%src : !iree.ref<!vmla.buffer>, %src_shape : !shape.ranked_shape<[4,?,8],i32>, %dst : !iree.ref<!vmla.buffer>) {
  // CHECK-DAG: %c1 = vm.const.i32 1 : i32
  // CHECK-DAG: %c2 = vm.const.i32 2 : i32
  // CHECK-DAG: %c4 = vm.const.i32 4 : i32
  // CHECK-DAG: %c8 = vm.const.i32 8 : i32
  // CHECK-NEXT: vm.call.variadic @vmla.transpose.x16(%arg0, [%c4, %arg1, %c8], [%c1, %c2], %arg2) : (!iree.ref<!vmla.buffer>, i32..., i32..., !iree.ref<!vmla.buffer>)
  "vmla.transpose"(%src, %src_shape, %dst) { dims = dense<[1, 2]> : tensor<2xi32>, element_type = i16 } : (!iree.ref<!vmla.buffer>, !shape.ranked_shape<[4,?,8],i32>, !iree.ref<!vmla.buffer>) -> ()
  return
}

// -----

// CHECK-LABEL: vm.func @convert
func @convert(%src : !iree.ref<!vmla.buffer>, %dst : !iree.ref<!vmla.buffer>) {
  // CHECK-NEXT:  vm.call @vmla.convert.f32.i8(%arg0, %arg1)
  "vmla.convert"(%src, %dst) { src_type = f32, dst_type = i8 } : (!iree.ref<!vmla.buffer>, !iree.ref<!vmla.buffer>) -> ()
  return
}

// -----

// CHECK-LABEL: vm.func @matmul
func @matmul(
    %lhs : !iree.ref<!vmla.buffer>,
    %lhs_shape : !shape.ranked_shape<[4,?],i32>,
    %rhs : !iree.ref<!vmla.buffer>,
    %rhs_shape : !shape.ranked_shape<[?,4],i32>,
    %dst : !iree.ref<!vmla.buffer>,
    %dst_shape : !shape.ranked_shape<[4,4],i32>) {
  // CHECK: vm.call.variadic @vmla.matmul.f32f32.f32(%arg0, [%c4, %arg1], %arg2, [%arg3, %c4_0], %arg4, [%c4_1, %c4_2])
  "vmla.matmul"(%lhs, %lhs_shape, %rhs, %rhs_shape, %dst, %dst_shape)
      { lhs_type = f32, rhs_type = f32, dst_type = f32 } :
      (!iree.ref<!vmla.buffer>,
       !shape.ranked_shape<[4,?],i32>,
       !iree.ref<!vmla.buffer>,
       !shape.ranked_shape<[?,4],i32>,
       !iree.ref<!vmla.buffer>,
       !shape.ranked_shape<[4,4],i32>) -> ()
  return
}
