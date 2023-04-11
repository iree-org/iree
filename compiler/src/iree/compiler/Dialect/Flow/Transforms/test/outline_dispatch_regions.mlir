// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-outline-dispatch-regions %s | FileCheck %s

//      CHECK: flow.executable private @staticShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.executable.export public @staticShapeDispatch_dispatch_0
//      CHECK: func.func @staticShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG:.+]]: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>,
// CHECK-SAME:     %[[RET:.+]]: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>) {
//  CHECK-DAG:   %[[ARG_VALUE:.+]] = flow.dispatch.tensor.load %[[ARG]], {{.*}} : !flow.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<8x4xf32>
// CHECK-NEXT:   %[[RET_VALUE:.+]] = "test.sink"(%[[ARG_VALUE]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
// CHECK-NEXT:   flow.dispatch.tensor.store %[[RET_VALUE]], %[[RET]], {{.*}} : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @staticShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func.func @staticShapeDispatch(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET:.+]] = flow.dispatch @staticShapeDispatch_dispatch_0::@staticShapeDispatch_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> tensor<4x8xf32> = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<8x4xf32>
    %ret_value = "test.sink"(%arg_value) : (tensor<8x4xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret,  offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
    flow.return
  }
  // CHECK-NEXT: return %[[RET]]
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_0
// CHECK-NEXT:   flow.executable.export public @dispatchFnMuli_dispatch_0
//      CHECK: func.func @dispatchFnMuli_dispatch_0(

//      CHECK: flow.executable private @dispatchFnMuli_dispatch_1
// CHECK-NEXT:   flow.executable.export public @dispatchFnMuli_dispatch_1
//      CHECK: func.func @dispatchFnMuli_dispatch_1(

// CHECK-LABEL: func.func @dispatchFnMuli(
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x4xf32>)
func.func @dispatchFnMuli(%arg0 : tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK-DAG: %[[X:.+]] = arith.constant 100
  %x = arith.constant 100 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 50
  %y = arith.constant 50 : index
  // CHECK: %[[RET0:.+]] = flow.dispatch @dispatchFnMuli_dispatch_0::@dispatchFnMuli_dispatch_0[
  // CHECK-SAME: %[[X]], %[[Y]]
  // CHECK-SAME: ](%[[ARG0]]) : (tensor<8x4xf32>) -> tensor<4x8xf32>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<8x4xf32>> -> tensor<8x4xf32>
    %ret_value = "test.sink1"(%arg_value) : (tensor<8x4xf32>) -> (tensor<4x8xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
    flow.return
  }
  // CHECK: %[[RET1:.+]] = flow.dispatch @dispatchFnMuli_dispatch_1::@dispatchFnMuli_dispatch_1[
  // CHECK-SAME: %[[Y]], %[[X]]
  // CHECK-SAME: ](%[[RET0]]) : (tensor<4x8xf32>) -> tensor<8x4xf32>
  %1 = flow.dispatch.workgroups[%y, %x](%0) : (tensor<4x8xf32>) -> (tensor<8x4xf32>) = (
    %arg: !flow.dispatch.tensor<readonly:tensor<4x8xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>
  ) {
    %arg_value = flow.dispatch.tensor.load %arg, offsets=[0, 0], sizes=[4, 8], strides=[1, 1] : !flow.dispatch.tensor<readonly:tensor<4x8xf32>> -> tensor<8x4xf32>
    %ret_value = "test.sink2"(%arg_value) : (tensor<8x4xf32>) -> (tensor<8x4xf32>)
    flow.dispatch.tensor.store %ret_value, %ret, offsets=[0, 0], sizes=[8, 4], strides=[1, 1] : tensor<8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x4xf32>>
    flow.return
  }
  // CHECK-NEXT: return %[[RET1]]
  return %1 : tensor<8x4xf32>
}

// -----

// CHECK: flow.executable private @dispatchFn1_dispatch_0

// CHECK-LABEL: func.func @dispatchFn1
func.func @dispatchFn1(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn1_dispatch_0::@dispatchFn1_dispatch_0
  // CHECK-SAME: stream.affinity = #hal.affinity.queue<[0]>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) attributes {
     stream.affinity = #hal.affinity.queue<[0]>
  } = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// CHECK: flow.executable private @dispatchFn2_dispatch_0

// CHECK-LABEL: func.func @dispatchFn2
func.func @dispatchFn2(%arg0 : tensor<8x4xf32>) -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  // CHECK: flow.dispatch @dispatchFn2_dispatch_0::@dispatchFn2_dispatch_0
  // CHECK-SAME: stream.affinity = #hal.affinity.queue<[1]>
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<8x4xf32>) -> (tensor<4x8xf32>) attributes {
    stream.affinity = #hal.affinity.queue<[1]>
  } = (
    %arg: !flow.dispatch.tensor<readonly:tensor<8x4xf32>>, %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

//      CHECK: flow.executable private @dynamicShapeDispatch_dispatch_0
// CHECK-NEXT:   flow.executable.export public @dynamicShapeDispatch_dispatch_0
//      CHECK: func.func @dynamicShapeDispatch_dispatch_0(
// CHECK-SAME:     %[[ARG_TENSOR:.+]]: !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>,
// CHECK-SAME:     %[[DIM1_CAPTURE:.+]]: index, %[[DIM3_CAPTURE:.+]]: index,
// CHECK-SAME:     %[[RET_TENSOR:.+]]: !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>) {

//      CHECK: %[[ARG_TILE:.+]] = flow.dispatch.tensor.load %[[ARG_TENSOR]], {{.+}} : !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>{%[[DIM1_CAPTURE]], %[[DIM3_CAPTURE]]}
// CHECK-NEXT: %[[RET_TILE:.+]] = "test.tile_math"(%[[ARG_TILE]])
// CHECK-NEXT: flow.dispatch.tensor.store %[[RET_TILE]], %[[RET_TENSOR]], {{.+}} -> !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>{%[[DIM3_CAPTURE]], %[[DIM1_CAPTURE]]}

// CHECK:   return
// CHECK-NEXT: }

// CHECK-LABEL: func.func @dynamicShapeDispatch(
// CHECK-SAME: %[[ARG0:.+]]: tensor<7x?x24x?xf32>
func.func @dynamicShapeDispatch(%arg0 : tensor<7x?x24x?xf32>) -> tensor<?x?x1024xf32> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  // CHECK-DAG: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %c1
  %dim1 = tensor.dim %arg0, %c1 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[DIM3:.+]] = tensor.dim %[[ARG0]], %c3
  %dim3 = tensor.dim %arg0, %c3 : tensor<7x?x24x?xf32>
  // CHECK-DAG: %[[X:.+]] = arith.constant 1024
  %x = arith.constant 1024 : index
  // CHECK-DAG: %[[Y:.+]] = arith.constant 512
  %y = arith.constant 512 : index
  // CHECK-NEXT: %[[RET0:.+]] = flow.dispatch @dynamicShapeDispatch_dispatch_0::@dynamicShapeDispatch_dispatch_0[
  // CHECK-SAME:   %[[X]], %[[Y]]
  // CHECK-SAME: ](%arg0, %[[DIM1]], %[[DIM3]])
  // CHECK-SAME: : (tensor<7x?x24x?xf32>{%[[DIM1]], %[[DIM3]]}, index, index) -> tensor<?x?x1024xf32>{%[[DIM3]], %[[DIM1]]}
  %ret0 = flow.dispatch.workgroups[%x, %y](%arg0, %dim1, %dim3) : (tensor<7x?x24x?xf32>{%dim1, %dim3}, index, index) -> tensor<?x?x1024xf32>{%dim3, %dim1} = (
    %arg: !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>,
    %dim1_capture: index, %dim3_capture: index,
    %ret: !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>
  ) {
    %arg_tile = flow.dispatch.tensor.load %arg, offsets=[0, 0, 0, 0], sizes=[7, %dim1_capture, 24, %dim3_capture], strides=[1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x?x24x?xf32>>{%dim1_capture, %dim3_capture} -> tensor<7x?x24x?xf32>
    %ret_tile = "test.tile_math"(%arg_tile) : (tensor<7x?x24x?xf32>) -> (tensor<?x?x1024xf32>)
    flow.dispatch.tensor.store %ret_tile, %ret, offsets=[0, 0, 0], sizes=[%dim3_capture, %dim1_capture, 1024], strides=[1, 1, 1] : tensor<?x?x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?x1024xf32>>{%dim3_capture, %dim1_capture}
    flow.return
  }
  // CHECK-NEXT: return %[[RET0]]
  return %ret0 : tensor<?x?x1024xf32>
}

// -----

// CHECK-LABEL: func.func @dispatchWithCountRegion
func.func @dispatchWithCountRegion(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  %0 = flow.dispatch.workgroups[%x, %y](%arg0) : (tensor<4xi32>) -> %arg0 =
      (%arg0_capture: !flow.dispatch.tensor<readwrite:tensor<4xi32>>) {
    flow.return
  } count(%x_capture: index, %y_capture: index) -> (index, index, index) {
    %z = arith.constant 1 : index
    flow.return %x_capture, %y_capture, %z : index, index, index
  }
  return %0 : tensor<4xi32>
}

// -----

// Dispatches containing some ops get a heuristics-driven summary in their name.

//      CHECK: flow.executable private @main_dispatch_0 {
// CHECK-NEXT:   flow.executable.export public @main_dispatch_0_fill_4x8
//      CHECK: func.func @main_dispatch_0_fill_4x8_f32(
func.func @main() -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  %0 = flow.dispatch.workgroups[%x, %y]() : () -> (tensor<4x8xf32>) = (
    %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    %cst = arith.constant 100.0 : f32
    %init = tensor.empty() : tensor<4x8xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x8xf32>) -> tensor<4x8xf32>
    flow.dispatch.tensor.store %fill, %ret, offsets = [0, 0], sizes = [4, 8], strides = [1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

// A cost model picks the "most expensive" op to include in the summary.

//      CHECK: flow.executable private @main_dispatch_0 {
// CHECK-NEXT:   flow.executable.export public @main_dispatch_0_fill_40
//      CHECK: func.func @main_dispatch_0_fill_40_f32(
func.func @main() -> tensor<10xf32> {
  %x = arith.constant 100 : index
  %0 = flow.dispatch.workgroups[%x]() : () -> (tensor<10xf32>) = (
    %ret: !flow.dispatch.tensor<writeonly:tensor<10xf32>>
  ) {
    %cst = arith.constant 100.0 : f32
    %init_small = tensor.empty() : tensor<10xf32>
    %fill_small = linalg.fill ins(%cst : f32) outs(%init_small : tensor<10xf32>) -> tensor<10xf32>
    // Note the ordering here - test that we don't just pick the first or the
    // last op. If an op in the middle has a higher cost then it should be used.
    %init_large = tensor.empty() : tensor<40xf32>
    %fill_large = linalg.fill ins(%cst : f32) outs(%init_large : tensor<40xf32>) -> tensor<40xf32>
    %init_medium = tensor.empty() : tensor<20xf32>
    %fill_medium = linalg.fill ins(%cst : f32) outs(%init_medium : tensor<20xf32>) -> tensor<20xf32>
    flow.dispatch.tensor.store %fill_small, %ret, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<writeonly:tensor<10xf32>>
    flow.return
  }
  return %0 : tensor<10xf32>
}

// -----

// Dynamic dimensions are considered the most expensive.

//      CHECK: flow.executable private @main_dispatch_0 {
// CHECK-NEXT:   flow.executable.export public @main_dispatch_0_fill_DxDxD
//      CHECK: func.func @main_dispatch_0_fill_DxDxD_f32(
func.func @main(%arg0 : index) -> tensor<10xf32> {
  %x = arith.constant 100 : index
  %0 = flow.dispatch.workgroups[%x]() : () -> (tensor<10xf32>) = (
    %arg0: index,
    %ret: !flow.dispatch.tensor<writeonly:tensor<10xf32>>
  ) {
    %cst = arith.constant 100.0 : f32
    %init_small = tensor.empty() : tensor<10xf32>
    %fill_small = linalg.fill ins(%cst : f32) outs(%init_small : tensor<10xf32>) -> tensor<10xf32>
    %init_dynamic = tensor.empty(%arg0, %arg0, %arg0) : tensor<?x?x?xf32>
    %fill_dynamic = linalg.fill ins(%cst : f32) outs(%init_dynamic : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    flow.dispatch.tensor.store %fill_small, %ret, offsets = [0], sizes = [10], strides = [1] : tensor<10xf32> -> !flow.dispatch.tensor<writeonly:tensor<10xf32>>
    flow.return
  }
  return %0 : tensor<10xf32>
}

// -----

// Dispatch key op with multiple datatypes should be reflected in summary.

//      CHECK: flow.executable private @main_dispatch_0 {
// CHECK-NEXT:   flow.executable.export public @main_dispatch_0_generic_4x8_i32xf32
//      CHECK: func.func @main_dispatch_0_generic_4x8_i32xf32(
func.func @main() -> tensor<4x8xf32> {
  %x = arith.constant 100 : index
  %y = arith.constant 50 : index
  %0 = flow.dispatch.workgroups[%x, %y]() : () -> (tensor<4x8xf32>) = (
    %ret: !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
  ) {
    %a = tensor.empty() : tensor<4x8xi32>
    %b = tensor.empty() : tensor<4x8xf32>
    %ans = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<4x8xi32>) outs(%b : tensor<4x8xf32>) {
      ^bb0(%b0 : i32, %b1 : f32):
        %1 = arith.index_cast %b0 : i32 to index
        %2 = tensor.extract %b[%1, %1] : tensor<4x8xf32>
        linalg.yield %2 : f32
      } -> tensor<4x8xf32>
    flow.dispatch.tensor.store %ans, %ret, offsets = [0, 0], sizes = [4, 8], strides = [1, 1] : tensor<4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<4x8xf32>>
    flow.return
  }
  return %0 : tensor<4x8xf32>
}

// -----

// Dispatches set_encoding and unset_encoding ops get a heuristics-driven
// summary in their name.

// CHECK: flow.executable private @main_dispatch_0
// CHECK:   func.func @main_dispatch_0_map_DxD
// CHECK: flow.executable private @main_dispatch_1
// CHECK:   func.func @main_dispatch_1_unset_encoding_MATMUL_F32F32F32_LHS_DxD
func.func @main(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: tensor<?x?xf32>, %arg4: index, %arg5: index) -> (tensor<?x?xf32>, index, index) {
  %0 = flow.tensor.tie_shape %arg0 : tensor<?x?xf32>{%arg1, %arg2}
  %1 = flow.tensor.tie_shape %arg3 : tensor<?x?xf32>{%arg4, %arg5}
  %2 = flow.dispatch.workgroups[%arg4, %arg5](%0, %1, %arg1, %arg2, %arg4, %arg5) : (tensor<?x?xf32>{%arg1, %arg2}, tensor<?x?xf32>{%arg4, %arg5}, index, index, index, index) -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>{%arg4, %arg5} =
      (%arg6: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg7: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>) {
    %4 = flow.dispatch.tie_shape %arg6 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg8, %arg9}
    %5 = flow.dispatch.tie_shape %arg7 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg10, %arg11}
    %6 = flow.dispatch.tie_shape %arg12 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%arg10, %arg11}
    %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%arg8, %arg9], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg8, %arg9} -> tensor<?x?xf32>
    %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%arg10, %arg11], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg10, %arg11} -> tensor<?x?xf32>
    %mapped = linalg.map { math.absf } ins(%7 : tensor<?x?xf32>) outs(%8 : tensor<?x?xf32>)
    %9 = iree_linalg_ext.set_encoding %mapped : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
    flow.dispatch.tensor.store %9, %6, offsets = [0, 0], sizes = [%arg10, %arg11], strides = [1, 1] : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%arg10, %arg11}
    flow.return
  } count(%arg6: index, %arg7: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_set_encoding_op %arg6, %arg7
    flow.return %x, %y, %z : index, index, index
  }
  %3 = flow.dispatch.workgroups[%arg4, %arg5](%2, %arg4, %arg5) : (tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>{%arg4, %arg5}, index, index) -> tensor<?x?xf32>{%arg4, %arg5} =
      (%arg6: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>, %arg7: index, %arg8: index, %arg9: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
    %4 = flow.dispatch.tie_shape %arg6 : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%arg7, %arg8}
    %5 = flow.dispatch.tie_shape %arg9 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg7, %arg8}
    %6 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%arg7, %arg8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>>{%arg7, %arg8} -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
    %7 = iree_linalg_ext.unset_encoding %6 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> -> tensor<?x?xf32>
    flow.dispatch.tensor.store %7, %5, offsets = [0, 0], sizes = [%arg7, %arg8], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg7, %arg8}
    flow.return
  } count(%arg6: index, %arg7: index) -> (index, index, index) {
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg6, %arg7
    flow.return %x, %y, %z : index, index, index
  }
  return %3, %arg1, %arg2 : tensor<?x?xf32>, index, index
}
