// RUN: iree-opt -split-input-file -iree-import-public %s | IreeFileCheck %s

// CHECK-LABEL: func @bv_func
// CHECK-SAME: (%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view)
// CHECK: return %arg0, %arg1 : !hal.buffer_view, !hal.buffer_view
builtin.func @bv_func(%arg0 : !iree.buffer_view, %arg1 : !iree.buffer_view) -> (!iree.buffer_view, !iree.buffer_view) {
  return %arg0, %arg1 : !iree.buffer_view, !iree.buffer_view
}

// -----
// CHECK-LABEL: func @list_func
// CHECK-SAME: (%arg0: !util.list<?>) -> !util.list<?>
builtin.func @list_func(%arg0 : !iree.list<!iree.variant>) -> !iree.list<!iree.variant> {
  return %arg0 : !iree.list<!iree.variant>
}

// -----
// CHECK-LABEL: func @ptr_func
// CHECK-SAME: (%arg0: !util.ptr<!hal.buffer_view>) -> !util.ptr<!hal.buffer_view>
builtin.func @ptr_func(%arg0 : !iree.ptr<!iree.buffer_view>) -> !iree.ptr<!iree.buffer_view> {
  return %arg0 : !iree.ptr<!iree.buffer_view>
}

// -----
// CHECK-LABEL: func @null_op
// CHECK: util.null : !util.variant
builtin.func @null_op() -> !iree.variant {
  %0 = iree.null : !iree.variant
  return %0 : !iree.variant
}

// -----
// CHECK-LABEL: func @tensor_to_buffer_view
// CHECK: hal.tensor.cast %arg0 : tensor<?x?x3xf32>{%arg1, %arg2} -> !hal.buffer_view
builtin.func @tensor_to_buffer_view(%arg0 : tensor<?x?x3xf32>, %arg1 : index, %arg2 : index) -> !iree.buffer_view {
  %0 = iree.cast.tensor_to_buffer_view %arg0 : tensor<?x?x3xf32> {%arg1, %arg2} -> !iree.buffer_view
  return %0 : !iree.buffer_view
}

// -----
// CHECK-LABEL: func @buffer_view_to_tensor
// CHECK: hal.tensor.cast %arg0 : !hal.buffer_view -> tensor<?x?x3xf32>{%arg1, %arg2}
builtin.func @tensor_to_buffer_view(%arg0 : !iree.buffer_view, %arg1 : index, %arg2 : index) -> tensor<?x?x3xf32> {
  %0 = iree.cast.buffer_view_to_tensor %arg0 : !iree.buffer_view -> tensor<?x?x3xf32> {%arg1, %arg2}
  return %0 : tensor<?x?x3xf32>
}

// TODO: Globals

// -----
// CHECK-LABEL: func @buffer_view_rank
// CHECK: hal.buffer_view.rank %arg0 : index
builtin.func @buffer_view_rank(%arg0 : !iree.buffer_view) -> index {
  %0 = iree.buffer_view.rank %arg0 : index
  return %0 : index
}

// -----
// CHECK-LABEL: func @buffer_view_dim
// CHECK: hal.buffer_view.dim %arg0, 0 : index
builtin.func @buffer_view_dim(%arg0 : !iree.buffer_view) -> index {
  %0 = iree.buffer_view.dim %arg0, 0 : index
  return %0: index
}

// TODO: iree.list.create
// TODO: iree.list.size
// TODO: iree.list.resize
// TODO: iree.list.get
// TODO: iree.list.set
// TODO: iree.tensor.reshape
// TODO: iree.tensor.load
// TODO: iree.tensor.store
// TODO: iree.tensor.splat
// TODO: iree.tensor.clone
// TODO: iree.tensor.slice
// TODO: iree.tensor.update
// TODO: iree.tensor.trace
