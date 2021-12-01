// RUN: iree-opt -split-input-file -iree-import-public %s | IreeFileCheck %s

// CHECK-LABEL: func @bv_func
// CHECK-SAME: (%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view)
// CHECK: return %arg0, %arg1 : !hal.buffer_view, !hal.buffer_view
builtin.func @bv_func(%arg0 : !iree_input.buffer_view, %arg1 : !iree_input.buffer_view) -> (!iree_input.buffer_view, !iree_input.buffer_view) {
  return %arg0, %arg1 : !iree_input.buffer_view, !iree_input.buffer_view
}

// -----
// CHECK-LABEL: func @list_func
// CHECK-SAME: (%arg0: !util.list<?>) -> !util.list<?>
builtin.func @list_func(%arg0 : !iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant> {
  return %arg0 : !iree_input.list<!iree_input.variant>
}

// -----
// CHECK-LABEL: func @list_func_call
// CHECK: call @list_func_call(%arg0) : (!util.list<?>) -> !util.list<?>
builtin.func @list_func_call(%arg0 : !iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant> {
  call @list_func_call(%arg0) : (!iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant>
  return %arg0 : !iree_input.list<!iree_input.variant>
}

// -----
// CHECK-LABEL: func @ptr_func
// CHECK-SAME: (%arg0: !util.ptr<!hal.buffer_view>) -> !util.ptr<!hal.buffer_view>
builtin.func @ptr_func(%arg0 : !iree_input.ptr<!iree_input.buffer_view>) -> !iree_input.ptr<!iree_input.buffer_view> {
  return %arg0 : !iree_input.ptr<!iree_input.buffer_view>
}

// -----
// CHECK-LABEL: func @null_op
// CHECK: util.null : !util.variant
builtin.func @null_op() -> !iree_input.variant {
  %0 = iree_input.null : !iree_input.variant
  return %0 : !iree_input.variant
}

// -----
// CHECK-LABEL: func @tensor_to_buffer_view
// CHECK: hal.tensor.export %arg0 : tensor<?x?x3xf32>{%arg1, %arg2} -> !hal.buffer_view
builtin.func @tensor_to_buffer_view(%arg0 : tensor<?x?x3xf32>, %arg1 : index, %arg2 : index) -> !iree_input.buffer_view {
  %0 = iree_input.cast.tensor_to_buffer_view %arg0 : tensor<?x?x3xf32>{%arg1, %arg2} -> !iree_input.buffer_view
  return %0 : !iree_input.buffer_view
}

// -----
// CHECK-LABEL: func @buffer_view_to_tensor
// CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x?x3xf32>{%arg1, %arg2}
builtin.func @buffer_view_to_tensor(%arg0 : !iree_input.buffer_view, %arg1 : index, %arg2 : index) -> tensor<?x?x3xf32> {
  %0 = iree_input.cast.buffer_view_to_tensor %arg0 : !iree_input.buffer_view -> tensor<?x?x3xf32>{%arg1, %arg2}
  return %0 : tensor<?x?x3xf32>
}

// -----
// CHECK-LABEL: func @buffer_view_rank
// CHECK: hal.buffer_view.rank<%arg0 : !hal.buffer_view> : index
builtin.func @buffer_view_rank(%arg0 : !iree_input.buffer_view) -> index {
  %0 = iree_input.buffer_view.rank %arg0 : index
  return %0 : index
}

// -----
// CHECK-LABEL: func @buffer_view_dim
// CHECK: hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
builtin.func @buffer_view_dim(%arg0 : !iree_input.buffer_view) -> index {
  %0 = iree_input.buffer_view.dim %arg0, 0 : index
  return %0: index
}

// -----
// CHECK-LABEL: func @list_create
// CHECK: util.list.create %arg0 : !util.list<?>
builtin.func @list_create(%arg0 : index) -> !iree_input.list<!iree_input.variant> {
  %0 = iree_input.list.create %arg0 : !iree_input.list<!iree_input.variant>
  return %0 : !iree_input.list<!iree_input.variant>
}

// -----
// CHECK-LABEL: func @list_size
// CHECK: util.list.size %arg0 : !util.list<?>
builtin.func @list_size(%arg0 : !iree_input.list<!iree_input.variant>) -> index {
  %0 = iree_input.list.size %arg0 : !iree_input.list<!iree_input.variant>
  return %0 : index
}

// -----
// CHECK-LABEL: func @list_resize
// CHECK: util.list.resize %arg0, %arg1 : !util.list<?>
builtin.func @list_resize(%arg0 : !iree_input.list<!iree_input.variant>, %arg1 : index) {
  iree_input.list.resize %arg0, %arg1 : !iree_input.list<!iree_input.variant>
  return
}

// -----
// CHECK-LABEL: func @list_get
// CHECK: util.list.get %arg0[%arg1] : !util.list<?>
builtin.func @list_get(%arg0 : !iree_input.list<!iree_input.variant>, %arg1 : index) -> !iree_input.variant {
  %0 = iree_input.list.get %arg0[%arg1] : !iree_input.list<!iree_input.variant> -> !iree_input.variant
  return %0 : !iree_input.variant
}

// -----
// CHECK-LABEL: func @list_set
// CHECK: util.list.set %arg0[%arg1], %arg2 : !util.list<?>
builtin.func @list_set(%arg0 : !iree_input.list<!iree_input.variant>, %arg1 : index, %arg2 : !iree_input.variant) {
  iree_input.list.set %arg0[%arg1], %arg2 : !iree_input.list<!iree_input.variant>, !iree_input.variant
  return
}

// -----
// CHECK-LABEL: func @tensor_reshape
// CHECK: flow.tensor.reshape %arg0 : tensor<?x?xf32>{%arg1, %arg2} -> tensor<?x?xf32>{%arg2, %arg1}
builtin.func @tensor_reshape(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %0 = iree_input.tensor.reshape %arg0 : tensor<?x?xf32>{%arg1, %arg2} -> tensor<?x?xf32>{%arg2, %arg1}
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func @tensor_load
// CHECK: flow.tensor.load %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
builtin.func @tensor_load(%arg0 : tensor<?x3xf32>, %arg1 : index, %arg2 : index, %arg3 : index) -> f32 {
  %0 = iree_input.tensor.load %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
  return %0 : f32
}

// -----
// CHECK-LABEL: func @tensor_store
// CHECK: flow.tensor.store %arg4, %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
builtin.func @tensor_store(%arg0 : tensor<?x3xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : f32) {
  iree_input.tensor.store %arg4, %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
  return
}

// -----
// CHECK-LABEL: func @tensor_splat
// CHECK: flow.tensor.splat %arg0 : tensor<?x?xf32>{%arg1, %arg2}
builtin.func @tensor_splat(%arg0 : f32, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %0 = iree_input.tensor.splat %arg0 : tensor<?x?xf32>{%arg1, %arg2}
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func @tensor_clone
// CHECK: flow.tensor.clone %arg0 : tensor<?x?xf32>{%arg1, %arg2}
builtin.func @tensor_clone(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %0 = iree_input.tensor.clone %arg0 : tensor<?x?xf32>{%arg1, %arg2}
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func @tensor_slice
// CHECK: flow.tensor.slice %arg0[%arg1 for %arg2] : tensor<?xf32>{%arg3} -> tensor<?xf32>{%arg4}
builtin.func @tensor_slice(%arg0 : tensor<?xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> tensor<?xf32> {
  %0 = iree_input.tensor.slice %arg0[%arg1 for %arg2] : tensor<?xf32>{%arg3} -> tensor<?xf32>{%arg4}
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: func @tensor_update
// CHECK: flow.tensor.update %arg3, %arg0[%arg1] : tensor<?xf32>{%arg2} -> %arg0 as tensor<?xf32>{%arg4}
builtin.func @tensor_update(%arg0 : tensor<?xf32>, %arg1 : index, %arg2 : index, %arg3 : tensor<?xf32>, %arg4 : index) -> tensor<?xf32> {
  %0 = iree_input.tensor.update %arg3, %arg0[%arg1] : tensor<?xf32>{%arg2} -> tensor<?xf32>{%arg4}
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: func @tensor_trace
// CHECK: flow.tensor.trace {key = "FOOBAR"} %arg0, %arg1 : tensor<5xf32>, tensor<3xf32>
builtin.func @tensor_trace(%arg0 : tensor<5xf32>, %arg1 : tensor<3xf32>) {
  iree_input.tensor.trace "FOOBAR" %arg0, %arg1 : tensor<5xf32>, tensor<3xf32>
  return
}

// -----
// CHECK-LABEL: module @globals
builtin.module @globals {
  // CHECK: util.global public mutable @global1 = 50 : i32
  iree_input.global mutable @global1 = 50 : i32
  // CHECK: util.global public mutable @global2 = 51 : i32
  iree_input.global public mutable @global2 = 51 : i32
  // CHECK: util.global private mutable @global3 = 52 : i32
  iree_input.global private mutable @global3 = 52 : i32
  // CHECK: util.global private @global4 = 53 : i32
  iree_input.global private @global4 = 53 : i32

  // CHECK: util.global public @global5 : tensor<4xi32>
  iree_input.global @global5 initializer(@initializer) : tensor<4xi32>
  // CHECK-NEXT: util.initializer {
  // CHECK-NEXT:   %[[VALUE:.+]] = call @initializer() : () -> tensor<4xi32>
  // CHECK-NEXT:   util.global.store %[[VALUE]], @global5 : tensor<4xi32>
  // CHECK-NEXT:   util.initializer.return
  // CHECK-NEXT: }
  // CHECK: func private @initializer() -> tensor<4xi32>
  builtin.func private @initializer() -> tensor<4xi32>
}

// -----
// CHECK-LABEL: module @global_load
builtin.module @global_load {
  iree_input.global private @v_loaded : tensor<4xi32>
  func @loaded() {
    // CHECK: util.global.load @v_loaded : tensor<4xi32>
    %0 = iree_input.global.load @v_loaded : tensor<4xi32>
    return
  }
}

// -----
// CHECK-LABEL: module @global_store
builtin.module @global_store {
  iree_input.global private mutable @v_stored : tensor<4xi32>
  func @stored() {
    // CHECK: %[[CST:.*]] = arith.constant
    %cst = arith.constant dense<5> : tensor<4xi32>
    // CHECK: util.global.store %[[CST]], @v_stored : tensor<4xi32>
    iree_input.global.store %cst, @v_stored : tensor<4xi32>
    return
  }
}

// -----
// CHECK-LABEL: module @global_load_indirect
builtin.module @global_load_indirect {
  iree_input.global private @v_loaded : tensor<4xf32>
  func @loaded_indirect() {
    // CHECK: %[[ADDR:.*]] = util.global.address @v_loaded : !util.ptr<tensor<4xf32>>
    %0 = iree_input.global.address @v_loaded : !iree_input.ptr<tensor<4xf32>>
    // CHECK: util.global.load.indirect %[[ADDR]] : !util.ptr<tensor<4xf32>> -> tensor<4xf32>
    %1 = iree_input.global.load.indirect %0 : !iree_input.ptr<tensor<4xf32>> -> tensor<4xf32>
    return
  }
}

// -----
// CHECK-LABEL: module @global_store_indirect
builtin.module @global_store_indirect {
  iree_input.global private mutable @v_stored : tensor<4xf32>
  func @stored_indirect(%arg0: tensor<4xf32>) {
    // CHECK: %[[ADDR:.*]] = util.global.address @v_stored : !util.ptr<tensor<4xf32>>
    %0 = iree_input.global.address @v_stored : !iree_input.ptr<tensor<4xf32>>
    // CHECK: util.global.store.indirect %arg0, %ptr_v_stored : tensor<4xf32> -> !util.ptr<tensor<4xf32>>
    iree_input.global.store.indirect %arg0, %0 : tensor<4xf32> -> !iree_input.ptr<tensor<4xf32>>
    return
  }
}
