// RUN: iree-opt --split-input-file %s | FileCheck %s

func.func @load_from_buffer(%arg0: memref<4xf32>) -> tensor<4xf32> {
  %value = iree_codegen.load_from_buffer %arg0 : memref<4xf32> -> tensor<4xf32>
  return %value : tensor<4xf32>
}
// CHECK-LABEL: func.func @load_from_buffer(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.load_from_buffer %[[ARG0]]
// CHECK-SAME:      : memref<4xf32> -> tensor<4xf32>

// -----

func.func @load_from_buffer_mixed_static_dynamic(%arg0: memref<?x4xf32>) -> tensor<4x?xf32> {
  %value = iree_codegen.load_from_buffer %arg0 : memref<?x4xf32> -> tensor<4x?xf32>
  return %value : tensor<4x?xf32>
}
// CHECK-LABEL: func.func @load_from_buffer_mixed_static_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.load_from_buffer %[[ARG0]]
// CHECK-SAME:      : memref<?x4xf32> -> tensor<4x?xf32>

// -----

func.func @load_from_strided_memref(
    %arg0: memref<?x?xf32, strided<[?, 1], offset: ?>>
) -> tensor<?x?xf32> {
  %value = iree_codegen.load_from_buffer %arg0
    : memref<?x?xf32, strided<[?, 1], offset: ?>> -> tensor<?x?xf32>
  return %value : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @load_from_strided_memref(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]:
// CHECK:         iree_codegen.load_from_buffer %[[ARG0]]
// CHECK-SAME:      : memref<?x?xf32, strided<[?, 1], offset: ?>> -> tensor<?x?xf32>

// -----

func.func @store_to_buffer(%arg0: tensor<4xf32>, %arg1: memref<4xf32>) {
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4xf32> into memref<4xf32>
  return
}
// CHECK-LABEL: func.func @store_to_buffer(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.store_to_buffer %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      : tensor<4xf32> into memref<4xf32>

// -----

func.func @store_to_buffer_mixed_static_dynamic(%arg0: tensor<4x?xf32>, %arg1: memref<?x4xf32>) {
  iree_codegen.store_to_buffer %arg0, %arg1 : tensor<4x?xf32> into memref<?x4xf32>
  return
}
// CHECK-LABEL: func.func @store_to_buffer_mixed_static_dynamic(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         iree_codegen.store_to_buffer %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      : tensor<4x?xf32> into memref<?x4xf32>

// -----

func.func @store_to_strided_memref(
    %arg0: tensor<?x?xf32>, %arg1: memref<?x?xf32, strided<[?, 1], offset: ?>>
) {
  iree_codegen.store_to_buffer %arg0, %arg1
    : tensor<?x?xf32> into memref<?x?xf32, strided<[?, 1], offset: ?>>
  return
}
// CHECK-LABEL: func.func @store_to_strided_memref(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]:
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]:
// CHECK:         iree_codegen.store_to_buffer %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      : tensor<?x?xf32> into memref<?x?xf32, strided<[?, 1], offset: ?>>

// -----

func.func @fusion_barrier(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = iree_codegen.fusion_barrier %arg0 : tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: func.func @fusion_barrier(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?xf32>
// CHECK:         iree_codegen.fusion_barrier %[[ARG0]] : tensor<?xf32>

// -----

func.func @index_hint(%idx: index) -> index {
  %hinted = iree_codegen.index_hint %idx([]) : index
  return %hinted : index
}
// CHECK-LABEL: func.func @index_hint(
// CHECK-SAME:    %[[IDX:[a-zA-Z0-9_]+]]: index
// CHECK:         %[[HINT:.+]] = iree_codegen.index_hint %[[IDX]]([]) : index
// CHECK:         return %[[HINT]]

// -----

// Test workgroup_scope attribute roundtrip.
func.func private @workgroup_scope_attr() attributes {
    scope = #iree_codegen.workgroup_scope
}
// CHECK-LABEL: func.func private @workgroup_scope_attr()
// CHECK-SAME:    scope = #iree_codegen.workgroup_scope

// -----

// Test workgroup_scope attribute with linearize option.
func.func private @workgroup_scope_attr_linearize() attributes {
    scope = #iree_codegen.workgroup_scope<linearize>
}
// CHECK-LABEL: func.func private @workgroup_scope_attr_linearize()
// CHECK-SAME:    scope = #iree_codegen.workgroup_scope<linearize>

// -----

// Test no pipeline attr inside translation_info.
func.func private @translation_info_no_pipeline() attributes {
    translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline>
}
// CHECK: #translation = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline>
// CHECK-LABEL: func.func private @translation_info_no_pipeline()
// CHECK-SAME:    translation_info = #translation

// -----

// Test transform dialect codegen pipeline attr inside translation_info.
func.func private @translation_info_transform_dialect_codegen() attributes {
    translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.transform_dialect_codegen codegen_spec = @__kernel_config workgroup_size = [64, 1, 1] subgroup_size = 32>
}
// CHECK: #translation = #iree_codegen.translation_info<pipeline = #iree_codegen.transform_dialect_codegen codegen_spec = @__kernel_config workgroup_size = [64, 1, 1] subgroup_size = 32>
// CHECK-LABEL: func.func private @translation_info_transform_dialect_codegen()
// CHECK-SAME:    translation_info = #translation

// -----

// Test VMVX pipeline attr inside translation_info.
func.func private @translation_info_vmvx_pipeline() attributes {
    translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.vmvx_pipeline>
}
// CHECK: #translation = #iree_codegen.translation_info<pipeline = #iree_codegen.vmvx_pipeline>
// CHECK-LABEL: func.func private @translation_info_vmvx_pipeline()
// CHECK-SAME:    translation_info = #translation

// -----

// Test constraints op with knobs and dims.
func.func @constraints_op(%arg0: index, %arg1: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
   knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">, #iree_codegen.smt.int_knob<"wg_n">]}
   dims(%arg0, %arg1) {
  ^bb0(%m: !smt.int, %n: !smt.int):
    %wg_m = iree_codegen.smt.knob "wg_m" : !smt.int
    %wg_n = iree_codegen.smt.knob "wg_n" : !smt.int
  }
  return
}
// CHECK-LABEL: func.func @constraints_op(
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9_]+]]: index
// CHECK:    iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK:     knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">, #iree_codegen.smt.int_knob<"wg_n">]}
// CHECK:     dims(%[[M]], %[[N]])
// CHECK:    ^bb0(%{{.*}}: !smt.int, %{{.*}}: !smt.int):
// CHECK:      iree_codegen.smt.knob "wg_m" : !smt.int
// CHECK:      iree_codegen.smt.knob "wg_n" : !smt.int

// -----

// Test constraints op with nested knobs (multiple dict groups) and SMT body.
func.func @constraints_op_with_smt_body(%arg0: index, %arg1: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
   knobs = {reduction = [#iree_codegen.smt.int_knob<"red_k">], workgroup = [#iree_codegen.smt.int_knob<"wg_m">, #iree_codegen.smt.int_knob<"wg_n">]}
   dims(%arg0, %arg1) {
  ^bb0(%m: !smt.int, %n: !smt.int):
    %wg_m = iree_codegen.smt.knob "wg_m" : !smt.int
    %wg_n = iree_codegen.smt.knob "wg_n" : !smt.int
    %red_k = iree_codegen.smt.knob "red_k" : !smt.int
    %zero = smt.int.constant 0
    %wg_m_pos = smt.int.cmp gt %wg_m, %zero
    smt.assert %wg_m_pos
  }
  return
}
// CHECK-LABEL: func.func @constraints_op_with_smt_body(
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[N:[a-zA-Z0-9_]+]]: index
// CHECK:    iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK:     knobs = {reduction = [#iree_codegen.smt.int_knob<"red_k">], workgroup = [#iree_codegen.smt.int_knob<"wg_m">, #iree_codegen.smt.int_knob<"wg_n">]}
// CHECK:     dims(%[[M]], %[[N]])
// CHECK:    ^bb0(%{{.*}}: !smt.int, %{{.*}}: !smt.int):
// CHECK:      iree_codegen.smt.knob "wg_m" : !smt.int
// CHECK:      iree_codegen.smt.knob "wg_n" : !smt.int
// CHECK:      iree_codegen.smt.knob "red_k" : !smt.int
// CHECK:      %[[ZERO:.*]] = smt.int.constant 0
// CHECK:      %[[CMP:.*]] = smt.int.cmp gt
// CHECK:      smt.assert %[[CMP]]

// -----

// Test assert op with static message (no format args).
func.func @assert_static_message(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {x = #iree_codegen.smt.int_knob<"x">}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %x = iree_codegen.smt.knob "x" : !smt.int
    %zero = smt.int.constant 0
    %cmp = smt.int.cmp gt %x, %zero
    iree_codegen.smt.assert %cmp, "x must be positive" : !smt.bool
  }
  return
}
// CHECK-LABEL: func.func @assert_static_message(
// CHECK:      iree_codegen.smt.assert %{{.*}}, "x must be positive" : !smt.bool

// -----

// Test assert op with format string args.
func.func @assert_with_format_args(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
   knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">, #iree_codegen.smt.int_knob<"wg_n">]}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %wg_m = iree_codegen.smt.knob "wg_m" : !smt.int
    %wg_n = iree_codegen.smt.knob "wg_n" : !smt.int
    %cmp = smt.int.cmp lt %wg_m, %wg_n
    iree_codegen.smt.assert %cmp, "wg_m ({}) < wg_n ({})", %wg_m, %wg_n : !smt.bool, !smt.int, !smt.int
  }
  return
}
// CHECK-LABEL: func.func @assert_with_format_args(
// CHECK:    ^bb0(%{{.*}}: !smt.int):
// CHECK:      %[[WG_M:.*]] = iree_codegen.smt.knob "wg_m"
// CHECK:      %[[WG_N:.*]] = iree_codegen.smt.knob "wg_n"
// CHECK:      %[[CMP:.*]] = smt.int.cmp lt %[[WG_M]], %[[WG_N]]
// CHECK:      iree_codegen.smt.assert %[[CMP]], "wg_m ({}) < wg_n ({})", %[[WG_M]], %[[WG_N]] : !smt.bool, !smt.int, !smt.int

// -----

// Test constraints op with empty dims.
func.func @constraints_op_empty_dims() {
  iree_codegen.smt.constraints target = <set = 1>, pipeline = #iree_codegen.no_pipeline,
   knobs = {}
   dims() {
  ^bb0:
  }
  return
}
// CHECK-LABEL: func.func @constraints_op_empty_dims(
// CHECK:    iree_codegen.smt.constraints target = <set = 1>, pipeline = #iree_codegen.no_pipeline,
// CHECK:     knobs = {}
// CHECK:     dims()

// Test constraints op with extra attributes (placed before the body).
func.func @constraints_op_with_attrs(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<TileAndFuse>,
   knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">]}
   dims(%arg0) attributes {some_tag = "hello"} {
  ^bb0(%m: !smt.int):
  }
  return
}
// CHECK-LABEL: func.func @constraints_op_with_attrs(
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]+]]: index
// CHECK:    iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<TileAndFuse>,
// CHECK:     knobs = {workgroup = [#iree_codegen.smt.int_knob<"wg_m">]}
// CHECK:     dims(%[[M]]) attributes {some_tag = "hello"}
// CHECK:    ^bb0(%{{.*}}: !smt.int):

// Test constraints op with PipelineAttrInterface (pass_pipeline attr).
func.func @constraints_op_with_pass_pipeline(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.pass_pipeline<"canonicalize">,
   knobs = {}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
  }
  return
}
// CHECK-LABEL: func.func @constraints_op_with_pass_pipeline(
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]+]]: index
// CHECK:    iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.pass_pipeline<"canonicalize">,
// CHECK:     knobs = {}
// CHECK:     dims(%[[M]])

// -----

// Test OneOfKnobAttr in constraints op knobs dict.
func.func @one_of_knob_attr(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {mma = #iree_codegen.smt.one_of_knob<"mma_idx", ["option_a", "option_b", "option_c"]>}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %idx = iree_codegen.smt.knob "mma_idx" : !smt.int
  }
  return
}
// CHECK-LABEL: func.func @one_of_knob_attr(
// CHECK:    knobs = {mma = #iree_codegen.smt.one_of_knob<"mma_idx", ["option_a", "option_b", "option_c"]>}

// -----

// Test OneOfKnobAttr with heterogeneous options (integer attrs).
func.func @one_of_knob_int_options(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {size = #iree_codegen.smt.one_of_knob<"size_idx", [16 : i64, 32 : i64, 64 : i64]>}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %idx = iree_codegen.smt.knob "size_idx" : !smt.int
  }
  return
}
// CHECK-LABEL: func.func @one_of_knob_int_options(
// CHECK:    knobs = {size = #iree_codegen.smt.one_of_knob<"size_idx", [16, 32, 64]>}

// -----

// Test smt.lookup op roundtrip.
func.func @smt_lookup(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {mma = #iree_codegen.smt.one_of_knob<"mma_idx", ["a", "b", "c"]>}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %idx = iree_codegen.smt.knob "mma_idx" : !smt.int
    %mma_m = iree_codegen.smt.lookup %idx [0, 1, 2] -> [16, 32, 64] : !smt.int
  }
  return
}
// CHECK-LABEL: func.func @smt_lookup(
// CHECK:    %[[IDX:.*]] = iree_codegen.smt.knob "mma_idx" : !smt.int
// CHECK:    iree_codegen.smt.lookup %[[IDX]] [0, 1, 2] -> [16, 32, 64] : !smt.int

// -----

// Test smt.lookup with non-contiguous keys not starting at 0.
func.func @smt_lookup_sparse(%arg0: index) {
  iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_codegen.no_pipeline,
   knobs = {mma = #iree_codegen.smt.one_of_knob<"mma_idx", ["a", "b", "c"]>}
   dims(%arg0) {
  ^bb0(%m: !smt.int):
    %idx = iree_codegen.smt.knob "mma_idx" : !smt.int
    %mma_m = iree_codegen.smt.lookup %idx [3, 7, 12] -> [16, 32, 64] : !smt.int
  }
  return
}
// CHECK-LABEL: func.func @smt_lookup_sparse(
// CHECK:    %[[IDX:.*]] = iree_codegen.smt.knob "mma_idx" : !smt.int
// CHECK:    iree_codegen.smt.lookup %[[IDX]] [3, 7, 12] -> [16, 32, 64] : !smt.int

// -----

iree_codegen.dispatch_config @matmul
    workgroup_size = [64, 16, 1] subgroup_size = 64 {
  ^bb0(%w0: index, %w1: index):
    %c1 = arith.constant 1 : index
    iree_codegen.yield %w0, %w1, %c1 : index, index, index
}
// CHECK-LABEL: iree_codegen.dispatch_config @matmul
// CHECK-SAME:    workgroup_size = [64, 16, 1]
// CHECK-SAME:    subgroup_size = 64
// CHECK:       ^bb0(%[[W0:.+]]: index, %[[W1:.+]]: index):
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         iree_codegen.yield %[[W0]], %[[W1]], %[[C1]] : index, index, index

// -----

iree_codegen.dispatch_config @no_subgroup
    workgroup_size = [256, 1, 1] {
  ^bb0(%w0: index):
    %c1 = arith.constant 1 : index
    iree_codegen.yield %w0, %c1, %c1 : index, index, index
}
// CHECK-LABEL: iree_codegen.dispatch_config @no_subgroup
// CHECK-SAME:    workgroup_size = [256, 1, 1]
// CHECK-NOT:     subgroup_size
// CHECK:       ^bb0(%[[W0:.+]]: index):
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         iree_codegen.yield %[[W0]], %[[C1]], %[[C1]] : index, index, index

// -----

iree_codegen.dispatch_config @static_count
    workgroup_size = [64] {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c4, %c2, %c1 : index, index, index
}
// CHECK-LABEL: iree_codegen.dispatch_config @static_count
// CHECK-SAME:    workgroup_size = [64]
// CHECK:         %[[C4:.+]] = arith.constant 4 : index
// CHECK:         %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         iree_codegen.yield %[[C4]], %[[C2]], %[[C1]] : index, index, index

// -----

iree_codegen.dispatch_config @no_config {
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  iree_codegen.yield %c4, %c2, %c1 : index, index, index
}
// CHECK-LABEL: iree_codegen.dispatch_config @no_config
// CHECK-NOT:     workgroup_size
// CHECK-NOT:     subgroup_size
// CHECK:         %[[C4:.+]] = arith.constant 4 : index
// CHECK:         %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         iree_codegen.yield %[[C4]], %[[C2]], %[[C1]] : index, index, index
