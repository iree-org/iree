// RUN: iree-opt --split-input-file --iree-import-public %s | FileCheck %s

// CHECK-LABEL: func.func @b_func
// CHECK-SAME: (%arg0: !hal.buffer, %arg1: !hal.buffer) -> (!hal.buffer, !hal.buffer)
// CHECK: return %arg0, %arg1 : !hal.buffer, !hal.buffer
func.func @b_func(%arg0 : !iree_input.buffer, %arg1 : !iree_input.buffer) -> (!iree_input.buffer, !iree_input.buffer) {
  return %arg0, %arg1 : !iree_input.buffer, !iree_input.buffer
}

// -----
// CHECK-LABEL: func.func @bv_func
// CHECK-SAME: (%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view)
// CHECK: return %arg0, %arg1 : !hal.buffer_view, !hal.buffer_view
func.func @bv_func(%arg0 : !iree_input.buffer_view, %arg1 : !iree_input.buffer_view) -> (!iree_input.buffer_view, !iree_input.buffer_view) {
  return %arg0, %arg1 : !iree_input.buffer_view, !iree_input.buffer_view
}

// -----
// CHECK-LABEL: func.func @list_func
// CHECK-SAME: (%arg0: !util.list<?>) -> !util.list<?>
func.func @list_func(%arg0 : !iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant> {
  return %arg0 : !iree_input.list<!iree_input.variant>
}

// -----
// CHECK-LABEL: func.func @list_func_retains_iree_abi
// CHECK-SAME: (%arg0: !util.list<?>) -> !util.list<?>
// CHECK-SAME: iree.abi = "FOOBAR"
func.func @list_func_retains_iree_abi(%arg0 : !iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant>
    attributes {iree.abi = "FOOBAR"} {
  return %arg0 : !iree_input.list<!iree_input.variant>
}

// -----
// CHECK-LABEL: func.func @list_func_call
// CHECK: call @list_func_call(%arg0) : (!util.list<?>) -> !util.list<?>
func.func @list_func_call(%arg0 : !iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant> {
  call @list_func_call(%arg0) : (!iree_input.list<!iree_input.variant>) -> !iree_input.list<!iree_input.variant>
  return %arg0 : !iree_input.list<!iree_input.variant>
}

// -----
// CHECK-LABEL: func.func @ptr_func
// CHECK-SAME: (%arg0: !util.ptr<!hal.buffer_view>) -> !util.ptr<!hal.buffer_view>
func.func @ptr_func(%arg0 : !iree_input.ptr<!iree_input.buffer_view>) -> !iree_input.ptr<!iree_input.buffer_view> {
  return %arg0 : !iree_input.ptr<!iree_input.buffer_view>
}

// -----
// CHECK-LABEL: func.func @null_op
// CHECK: util.null : !util.variant
func.func @null_op() -> !iree_input.variant {
  %0 = iree_input.null : !iree_input.variant
  return %0 : !iree_input.variant
}

//----
// CHECK-LABEL: func.func @buffer_subspan
// CHECK-SAME: (%arg0: !hal.buffer) -> !hal.buffer
// CHECK: %[[OFFSET:.+]] = arith.constant 100
// CHECK: %[[LENGTH:.+]] = arith.constant 200
// CHECK: %buffer = hal.buffer.subspan
// CHECK-SAME: <%arg0 : !hal.buffer>[%[[OFFSET]], %[[LENGTH]]] : !hal.buffer

func.func @buffer_subspan(%arg0: !iree_input.buffer) -> !iree_input.buffer {
  %offset = arith.constant 100 : index
  %length = arith.constant 200 : index
  %buffer = iree_input.buffer.subspan<%arg0 : !iree_input.buffer>[%offset, %length] : !iree_input.buffer
  return %buffer : !iree_input.buffer
}

//----
// CHECK-LABEL: func.func @buffer_view_create
// CHECK-SAME: (%arg0: !hal.buffer) -> !hal.buffer_view
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[C32:.*]] = arith.constant 32 : i32
// CHECK: %view = hal.buffer_view.create
// CHECK-SAME: buffer(%arg0 : !hal.buffer)[%[[C0]], %[[C128]]]
// CHECK-SAME: shape([%[[C2]], %[[C64]]])
// CHECK-SAME: type(%[[C32]])
// CHECK-SAME: encoding(%[[C1]]) : !hal.buffer_view

func.func @buffer_view_create(%arg0: !iree_input.buffer) -> !iree_input.buffer_view {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %view = iree_input.buffer_view.create buffer(%arg0 : !iree_input.buffer)[%c0, %c128]
                                        shape([%c2, %c64])
                                        type(%c32_i32)
                                        encoding(%c1_i32) : !iree_input.buffer_view
  return %view : !iree_input.buffer_view
}

// -----
// CHECK-LABEL: func.func @tensor_export
// CHECK: hal.tensor.export %arg0 : tensor<?x?x3xf32>{%arg1, %arg2} -> !hal.buffer_view
func.func @tensor_export(%arg0 : tensor<?x?x3xf32>, %arg1 : index, %arg2 : index) -> !iree_input.buffer_view {
  %0 = iree_input.tensor.export %arg0 : tensor<?x?x3xf32>{%arg1, %arg2} -> !iree_input.buffer_view
  return %0 : !iree_input.buffer_view
}

// -----
// CHECK-LABEL: func.func @tensor_export_static
// CHECK: hal.tensor.export %arg0 : tensor<3xf32> -> !hal.buffer_view
func.func @tensor_export_static(%arg0 : tensor<3xf32>) -> !iree_input.buffer_view {
  %0 = iree_input.tensor.export %arg0 : tensor<3xf32> -> !iree_input.buffer_view
  return %0 : !iree_input.buffer_view
}

// -----
// CHECK-LABEL: func.func @tensor_export_implicit_dims
// CHECK: %[[ZERO:.*]] = arith.constant 0
// CHECK: %[[D0:.*]] = tensor.dim %arg0, %[[ZERO]]
// CHECK: %[[ONE:.*]] = arith.constant 1
// CHECK: %[[D1:.*]] = tensor.dim %arg0, %[[ONE]]
// CHECK: hal.tensor.export %arg0 : tensor<?x?x3xf32>{%[[D0]], %[[D1]]} -> !hal.buffer_view
func.func @tensor_export_implicit_dims(%arg0 : tensor<?x?x3xf32>) -> !iree_input.buffer_view {
  %0 = iree_input.tensor.export %arg0 : tensor<?x?x3xf32> -> !iree_input.buffer_view
  return %0 : !iree_input.buffer_view
}

// -----
// CHECK-LABEL: func.func @tensor_import
// CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x?x3xf32>{%arg1, %arg2}
func.func @tensor_import(%arg0 : !iree_input.buffer_view, %arg1 : index, %arg2 : index) -> tensor<?x?x3xf32> {
  %0 = iree_input.tensor.import %arg0 : !iree_input.buffer_view -> tensor<?x?x3xf32>{%arg1, %arg2}
  return %0 : tensor<?x?x3xf32>
}

// -----
// CHECK-LABEL: func.func @tensor_import_static
// CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<3xf32>
func.func @tensor_import_static(%arg0 : !iree_input.buffer_view) -> tensor<3xf32> {
  %0 = iree_input.tensor.import %arg0 : !iree_input.buffer_view -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----
// CHECK-LABEL: func.func @tensor_import_implicit_dims
// CHECK: %[[D0:.*]] = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
// CHECK: %[[D1:.*]] = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
// CHECK: hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?x?x3xf32>{%[[D0]], %[[D1]]}
func.func @tensor_import_implicit_dims(%arg0 : !iree_input.buffer_view) -> tensor<?x?x3xf32> {
  %0 = iree_input.tensor.import %arg0 : !iree_input.buffer_view -> tensor<?x?x3xf32>
  return %0 : tensor<?x?x3xf32>
}

// -----
// CHECK-LABEL: func.func @buffer_view_rank
// CHECK: hal.buffer_view.rank<%arg0 : !hal.buffer_view> : index
func.func @buffer_view_rank(%arg0 : !iree_input.buffer_view) -> index {
  %0 = iree_input.buffer_view.rank %arg0 : index
  return %0 : index
}

// -----
// CHECK-LABEL: func.func @byte_buffer_constant
// CHECK: %[[B:.*]] = util.buffer.constant "name" {alignment = 64 : index, mime_type = "text/plain"} : !util.buffer = "foo"
// CHECK: return %[[B]] : !util.buffer
func.func @byte_buffer_constant() -> !iree_input.byte_buffer {
  %0 = iree_input.byte_buffer.constant "name" {alignment = 64 : index, mime_type = "text/plain"} : !iree_input.byte_buffer = "foo"
  return %0 : !iree_input.byte_buffer
}

// -----
// CHECK-LABEL: func.func @buffer_view_dim
// CHECK: hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
func.func @buffer_view_dim(%arg0 : !iree_input.buffer_view) -> index {
  %0 = iree_input.buffer_view.dim %arg0, 0 : index
  return %0: index
}

// -----
// CHECK-LABEL: func.func @list_create
// CHECK: util.list.create %arg0 : !util.list<?>
func.func @list_create(%arg0 : index) -> !iree_input.list<!iree_input.variant> {
  %0 = iree_input.list.create %arg0 : !iree_input.list<!iree_input.variant>
  return %0 : !iree_input.list<!iree_input.variant>
}

// -----
// CHECK-LABEL: func.func @list_size
// CHECK: util.list.size %arg0 : !util.list<?>
func.func @list_size(%arg0 : !iree_input.list<!iree_input.variant>) -> index {
  %0 = iree_input.list.size %arg0 : !iree_input.list<!iree_input.variant>
  return %0 : index
}

// -----
// CHECK-LABEL: func.func @list_resize
// CHECK: util.list.resize %arg0, %arg1 : !util.list<?>
func.func @list_resize(%arg0 : !iree_input.list<!iree_input.variant>, %arg1 : index) {
  iree_input.list.resize %arg0, %arg1 : !iree_input.list<!iree_input.variant>
  return
}

// -----
// CHECK-LABEL: func.func @list_get
// CHECK: util.list.get %arg0[%arg1] : !util.list<?>
func.func @list_get(%arg0 : !iree_input.list<!iree_input.variant>, %arg1 : index) -> !iree_input.variant {
  %0 = iree_input.list.get %arg0[%arg1] : !iree_input.list<!iree_input.variant> -> !iree_input.variant
  return %0 : !iree_input.variant
}

// -----
// CHECK-LABEL: func.func @list_set
// CHECK: util.list.set %arg0[%arg1], %arg2 : !util.list<?>
func.func @list_set(%arg0 : !iree_input.list<!iree_input.variant>, %arg1 : index, %arg2 : !iree_input.variant) {
  iree_input.list.set %arg0[%arg1], %arg2 : !iree_input.list<!iree_input.variant>, !iree_input.variant
  return
}

// -----
// CHECK-LABEL: func.func @tensor_reshape
// CHECK: flow.tensor.reshape %arg0 : tensor<?x?xf32>{%arg1, %arg2} -> tensor<?x?xf32>{%arg2, %arg1}
func.func @tensor_reshape(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %0 = iree_input.tensor.reshape %arg0 : tensor<?x?xf32>{%arg1, %arg2} -> tensor<?x?xf32>{%arg2, %arg1}
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func.func @tensor_load
// CHECK: flow.tensor.load %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
func.func @tensor_load(%arg0 : tensor<?x3xf32>, %arg1 : index, %arg2 : index, %arg3 : index) -> f32 {
  %0 = iree_input.tensor.load %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
  return %0 : f32
}

// -----
// CHECK-LABEL: func.func @tensor_store
// CHECK: flow.tensor.store %arg4, %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
func.func @tensor_store(%arg0 : tensor<?x3xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : f32) {
  iree_input.tensor.store %arg4, %arg0[%arg2, %arg3] : tensor<?x3xf32>{%arg1}
  return
}

// -----
// CHECK-LABEL: func.func @tensor_splat
// CHECK: flow.tensor.splat %arg0 : tensor<?x?xf32>{%arg1, %arg2}
func.func @tensor_splat(%arg0 : f32, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %0 = iree_input.tensor.splat %arg0 : tensor<?x?xf32>{%arg1, %arg2}
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func.func @tensor_clone
// CHECK: flow.tensor.clone %arg0 : tensor<?x?xf32>{%arg1, %arg2}
func.func @tensor_clone(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index) -> tensor<?x?xf32> {
  %0 = iree_input.tensor.clone %arg0 : tensor<?x?xf32>{%arg1, %arg2}
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: func.func @tensor_slice
// CHECK: flow.tensor.slice %arg0[%arg1 for %arg2] : tensor<?xf32>{%arg3} -> tensor<?xf32>{%arg4}
func.func @tensor_slice(%arg0 : tensor<?xf32>, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index) -> tensor<?xf32> {
  %0 = iree_input.tensor.slice %arg0[%arg1 for %arg2] : tensor<?xf32>{%arg3} -> tensor<?xf32>{%arg4}
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: func.func @tensor_update
// CHECK: flow.tensor.update %arg3, %arg0[%arg1] : tensor<?xf32>{%arg2} -> %arg0 as tensor<?xf32>{%arg4}
func.func @tensor_update(%arg0 : tensor<?xf32>, %arg1 : index, %arg2 : index, %arg3 : tensor<?xf32>, %arg4 : index) -> tensor<?xf32> {
  %0 = iree_input.tensor.update %arg3, %arg0[%arg1] : tensor<?xf32>{%arg2} -> %arg0 as tensor<?xf32>{%arg4}
  return %0 : tensor<?xf32>
}

// -----
// CHECK-LABEL: func.func @tensor_trace
//      CHECK: flow.tensor.trace "FOOBAR" = [
// CHECK-SAME:   %arg0 : tensor<5xf32>,
// CHECK-SAME:   %arg1 : tensor<?x3xf32>{%arg2}
// CHECK-SAME: ]
func.func @tensor_trace(%arg0: tensor<5xf32>, %arg1: tensor<?x3xf32>, %arg2: index) {
  iree_input.tensor.trace "FOOBAR" = [
    %arg0 : tensor<5xf32>,
    %arg1 : tensor<?x3xf32>{%arg2}
  ]
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
  // CHECK-NEXT:   %[[VALUE:.+]] = func.call @initializer() : () -> tensor<4xi32>
  // CHECK-NEXT:   util.global.store %[[VALUE]], @global5 : tensor<4xi32>
  // CHECK-NEXT:   util.initializer.return
  // CHECK-NEXT: }
  // CHECK: func.func private @initializer() -> tensor<4xi32>
  func.func private @initializer() -> tensor<4xi32>
}

// -----
// CHECK-LABEL: module @global_load
builtin.module @global_load {
  iree_input.global private @v_loaded : tensor<4xi32>
  func.func @loaded() {
    // CHECK: util.global.load @v_loaded : tensor<4xi32>
    %0 = iree_input.global.load @v_loaded : tensor<4xi32>
    return
  }
}

// -----
// CHECK-LABEL: module @global_store
builtin.module @global_store {
  iree_input.global private mutable @v_stored : tensor<4xi32>
  func.func @stored() {
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
  func.func @loaded_indirect() {
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
  func.func @stored_indirect(%arg0: tensor<4xf32>) {
    // CHECK: %[[ADDR:.*]] = util.global.address @v_stored : !util.ptr<tensor<4xf32>>
    %0 = iree_input.global.address @v_stored : !iree_input.ptr<tensor<4xf32>>
    // CHECK: util.global.store.indirect %arg0, %ptr_v_stored : tensor<4xf32> -> !util.ptr<tensor<4xf32>>
    iree_input.global.store.indirect %arg0, %0 : tensor<4xf32> -> !iree_input.ptr<tensor<4xf32>>
    return
  }
}

// -----
// CHECK-LABEL: func.func @optimization_barrier
// CHECK: util.optimization_barrier %arg0 : tensor<f32>
func.func @optimization_barrier(%arg0 : tensor<f32>) -> tensor<f32> {
  %0 = iree_input.optimization_barrier %arg0 : tensor<f32>
  return %0 : tensor<f32>
}

// -----
// CHECK: #[[SM75:.*]] = #hal.executable.target<"cuda",
// CHECK-SAME:           "cuda-nvptx-fb", {target_arch = "sm_75"}>

// CHECK: #[[LAYOUT:.*]] = #hal.pipeline.layout<push_constants = 1,
// CHECK-SAME: sets = [<0, bindings = [
// CHECK-SAME:             <0, storage_buffer, ReadOnly>,
// CHECK-SAME:             <1, storage_buffer>
// CHECK-SAME: ]>]>

// CHECK: hal.executable.source private @executable
// CHECK-SAME: {objects = #hal.executable.objects<{
// CHECK-SAME: #[[SM75]] = [#hal.executable.object<{path = "executable.ptx"}>]

// CHECK: hal.executable.export public @add ordinal(0)
// CHECL-SAME: layout(#[[LAYOUT]])
// CHECK-SAME: workgroup_size = [64 : index, 1 : index, 1 : index]

// CHECK: flow.dispatch @executable::@add[%c0](%arg0, %arg1) : {{.*}} -> %arg1

#sm75 = #iree_input.executable.target<"cuda", "cuda-nvptx-fb", {
  target_arch = "sm_75"
}>

builtin.module @executable_source {

  iree_input.executable.source private @executable attributes {
    objects = #iree_input.executable.objects<{
      #sm75 = [#iree_input.executable.object<{path = "executable.ptx"}>]
    }>
  } {
    iree_input.executable.export public @add ordinal(0)
      layout(#iree_input.pipeline.layout<push_constants = 1, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer>
        ]>
      ]>) attributes {
      workgroup_size = [64 : index, 1 : index, 1 : index]
    }
  }

  func.func @dispatch(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %c0 = arith.constant 0 : index
    %0 = flow.dispatch @executable::@add[%c0](%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> %arg1
    return %0 : tensor<f32>
  }
}
