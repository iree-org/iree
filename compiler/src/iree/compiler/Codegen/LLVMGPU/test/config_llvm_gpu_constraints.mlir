// RUN: iree-opt --split-input-file \
// RUN:   --iree-codegen-experimental-verify-pipeline-constraints \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-codegen-insert-smt-constraints))' %s \
// RUN:   | FileCheck %s

#gpu_target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target}>

func.func @fill_root_op()
    attributes {hal.executable.target = #exec_target} {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @fill_root_op
// CHECK:       linalg.fill {{.+}} #iree_codegen.root_op<set = 0>
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   knobs

func.func @matmul_and_fill() attributes {hal.executable.target = #exec_target} {
  %cst = arith.constant 0.0 : f32
  %lhs = tensor.empty() : tensor<128x64xf32>
  %rhs = tensor.empty() : tensor<64x256xf32>
  %empty = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>}
      ins(%cst : f32)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul_and_fill
// CHECK:       linalg.fill {{.+}} #iree_codegen.root_op<set = 0>
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   knobs
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = [[SET:[0-9]+]]>
// CHECK:       iree_codegen.smt.constraints target = <set = [[SET]]>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK-NEXT:  knobs = {
// CHECK-DAG:   mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>]>
// CHECK-DAG:   reduction = [0, 0, #iree_codegen.smt.int_knob<"red_2">]
// CHECK-DAG{LITERAL}: subgroup_basis = [[#iree_codegen.smt.int_knob<"sg_m_cnt">, #iree_codegen.smt.int_knob<"sg_n_cnt">, 1], [0, 1, 2]]
// CHECK-DAG:   subgroup_size = #iree_codegen.smt.int_knob<"sg_size">
// CHECK-DAG:   workgroup = [#iree_codegen.smt.int_knob<"wg_0">, #iree_codegen.smt.int_knob<"wg_1">, 0]
// CHECK-DAG:   workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, #iree_codegen.smt.int_knob<"wg_size_y">, #iree_codegen.smt.int_knob<"wg_size_z">]
// CHECK-SAME:  }
// CHECK:       "dim_0 must be divisible by wg_0 ({} % {} == 0)"
// CHECK:       "dim_1 must be divisible by wg_1 ({} % {} == 0)"
// CHECK:       "dim_2 must be divisible by red_2 ({} % {} == 0)"
// CHECK-NOT:   "dim_{{[0-9]+}} must be divisible by {{.*}}"

func.func @conv_pooling_nhwc_sum_root_op()
    attributes {hal.executable.target = #exec_target} {
  %cst = arith.constant 0.0 : f32
  %input = tensor.empty() : tensor<1x18x18x64xf32>
  %window = tensor.empty() : tensor<3x3xf32>
  %empty = tensor.empty() : tensor<1x16x16x64xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 2>} ins(%cst : f32)
      outs(%empty : tensor<1x16x16x64xf32>) -> tensor<1x16x16x64xf32>
  %result = linalg.pooling_nhwc_sum {
      dilations = dense<1> : vector<2xi64>,
      root_op = #iree_codegen.root_op<set = 2>,
      strides = dense<1> : vector<2xi64>}
      ins(%input, %window : tensor<1x18x18x64xf32>, tensor<3x3xf32>)
      outs(%fill : tensor<1x16x16x64xf32>) -> tensor<1x16x16x64xf32>
  return
}

// CHECK-LABEL: func.func @conv_pooling_nhwc_sum
// CHECK:       linalg.pooling_nhwc_sum
// CHECK-SAME:  #iree_codegen.root_op<set = 2>
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   knobs

func.func @conv_2d_nhwc_hwcf()
    attributes {hal.executable.target = #exec_target} {
  %cst = arith.constant 0.0 : f32
  %input = tensor.empty() : tensor<1x18x18x64xf32>
  %filter = tensor.empty() : tensor<3x3x64x128xf32>
  %empty = tensor.empty() : tensor<1x16x16x128xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 1>} ins(%cst : f32)
      outs(%empty : tensor<1x16x16x128xf32>) -> tensor<1x16x16x128xf32>
  %result = linalg.conv_2d_nhwc_hwcf {
      dilations = dense<1> : tensor<2xi64>,
      root_op = #iree_codegen.root_op<set = 1>,
      strides = dense<1> : tensor<2xi64>}
      ins(%input, %filter : tensor<1x18x18x64xf32>,
                              tensor<3x3x64x128xf32>)
      outs(%fill : tensor<1x16x16x128xf32>) -> tensor<1x16x16x128xf32>
  return
}

// CHECK-LABEL: func.func @conv_2d_nhwc_hwcf
// CHECK:       linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:  #iree_codegen.root_op<set = 1>
// CHECK:       iree_codegen.smt.constraints target = <set = 1>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK-NEXT:  knobs = {
// CHECK-DAG:   mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>]>
// CHECK-DAG:   reduction = [0, 0, 0, 0, 1, 1, #iree_codegen.smt.int_knob<"red_6">]
// CHECK-DAG{LITERAL}: subgroup_basis = [[1, 1, #iree_codegen.smt.int_knob<"sg_m_cnt">, #iree_codegen.smt.int_knob<"sg_n_cnt">, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6]]
// CHECK-DAG:   subgroup_size = #iree_codegen.smt.int_knob<"sg_size">
// CHECK-DAG:   workgroup = [1, 1, #iree_codegen.smt.int_knob<"wg_2">, #iree_codegen.smt.int_knob<"wg_3">, 0, 0, 0]
// CHECK-DAG:   workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, #iree_codegen.smt.int_knob<"wg_size_y">, #iree_codegen.smt.int_knob<"wg_size_z">]
// CHECK-SAME:  }
// CHECK:       "dim_2 must be divisible by wg_2 ({} % {} == 0)"
// CHECK:       "dim_3 must be divisible by wg_3 ({} % {} == 0)"
// CHECK:       "dim_6 must be divisible by red_6 ({} % {} == 0)"
// CHECK-NOT:   "dim_{{[0-9]+}} must be divisible by {{.*}}"

#map_lhs = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map_rhs = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
#map_out = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

func.func @expanded_matmul()
    attributes {hal.executable.target = #exec_target} {
  %cst = arith.constant 0.0 : f32
  %lhs = tensor.empty() : tensor<2x64x2048xf32>
  %rhs = tensor.empty() : tensor<10x64x2048xf32>
  %empty = tensor.empty() : tensor<2x10x64x64xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>} ins(%cst : f32)
      outs(%empty : tensor<2x10x64x64xf32>) -> tensor<2x10x64x64xf32>
  %result = linalg.generic {
      indexing_maps = [#map_lhs, #map_rhs, #map_out],
      iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction"],
      root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<2x64x2048xf32>,
                       tensor<10x64x2048xf32>)
      outs(%fill : tensor<2x10x64x64xf32>) {
  ^bb0(%in_lhs: f32, %in_rhs: f32, %out: f32):
    %mul = arith.mulf %in_lhs, %in_rhs : f32
    %add = arith.addf %mul, %out : f32
    linalg.yield %add : f32
  } -> tensor<2x10x64x64xf32>
  return
}

// CHECK-LABEL: func.func @expanded_matmul
// CHECK:       linalg.generic
// CHECK:       iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK-NEXT:  knobs = {
// CHECK-DAG:   mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>]>
// CHECK-DAG:   reduction = [0, 0, 0, 0, #iree_codegen.smt.int_knob<"red_4">]
// CHECK-DAG{LITERAL}: subgroup_basis = [[1, 1, #iree_codegen.smt.int_knob<"sg_m_cnt">, #iree_codegen.smt.int_knob<"sg_n_cnt">, 1], [0, 1, 2, 3, 4]]
// CHECK-DAG:   subgroup_size = #iree_codegen.smt.int_knob<"sg_size">
// CHECK-DAG:   workgroup = [1, 1, #iree_codegen.smt.int_knob<"wg_2">, #iree_codegen.smt.int_knob<"wg_3">, 0]
// CHECK-DAG:   workgroup_size = [#iree_codegen.smt.int_knob<"wg_size_x">, #iree_codegen.smt.int_knob<"wg_size_y">, #iree_codegen.smt.int_knob<"wg_size_z">]
// CHECK-SAME:  }
// CHECK:       "dim_2 must be divisible by wg_2 ({} % {} == 0)"
// CHECK:       "dim_3 must be divisible by wg_3 ({} % {} == 0)"
// CHECK:       "dim_4 must be divisible by red_4 ({} % {} == 0)"
// CHECK-NOT:   "dim_{{[0-9]+}} must be divisible by {{.*}}"

func.func @matmul_dynamic_shapes(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>,
                          %empty: tensor<?x?xf32>) -> tensor<?x?xf32>
    attributes {hal.executable.target = #exec_target} {
  %r = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @matmul_dynamic_shapes
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = 0>
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   knobs

func.func @matmul_with_mismatch_mma_element_types()
    attributes {hal.executable.target = #exec_target} {
  %lhs = tensor.empty() : tensor<128x64xf16>
  %rhs = tensor.empty() : tensor<64x256xf16>
  %empty = tensor.empty() : tensor<128x256xf16>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x256xf16>)
      outs(%empty : tensor<128x256xf16>) -> tensor<128x256xf16>
  return
}

// CHECK-LABEL: func.func @matmul_with_mismatch_mma_element_types
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = 0>
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   knobs

// -----

#gpu_target_no_mmas = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [],
  subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_no_mmas = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_no_mmas}>

func.func @matmul_with_no_compatible_mmas()
    attributes {hal.executable.target = #exec_target_no_mmas} {
  %lhs = tensor.empty() : tensor<128x64xf32>
  %rhs = tensor.empty() : tensor<64x256xf32>
  %empty = tensor.empty() : tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul_with_no_compatible_mmas
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = 0>
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   knobs

// -----

#gpu_target_mismatch_mma_subgroup_size = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [32],
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_mismatch_mma_subgroup_size = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_mismatch_mma_subgroup_size}>

func.func @matmul_with_mismatch_mma_subgroup_size()
    attributes {hal.executable.target = #exec_target_mismatch_mma_subgroup_size} {
  %lhs = tensor.empty() : tensor<128x64xf32>
  %rhs = tensor.empty() : tensor<64x256xf32>
  %empty = tensor.empty() : tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul_with_mismatch_mma_subgroup_size
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = 0>
// CHECK-NOT:   iree_codegen.smt.constraints
// CHECK-NOT:   knobs

// -----

#gpu_target_multiple_compatible_mmas = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>,
         <MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x4_F32>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_multiple_compatible_mmas = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_multiple_compatible_mmas}>

func.func @matmul_with_multiple_compatible_mmas()
    attributes {hal.executable.target = #exec_target_multiple_compatible_mmas} {
  %cst = arith.constant 0.0 : f32
  %lhs = tensor.empty() : tensor<128x64xf16>
  %rhs = tensor.empty() : tensor<64x256xf16>
  %empty = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>} ins(%cst : f32)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x256xf16>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul_with_multiple_compatible_mmas
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = 0>
// CHECK:       iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK:       mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>]>

func.func @matmul_with_duplicate_mmas_deduped()
    attributes {hal.executable.target = #exec_target_multiple_compatible_mmas} {
  %lhs = tensor.empty() : tensor<128x64xf32>
  %rhs = tensor.empty() : tensor<64x256xf32>
  %empty = tensor.empty() : tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 1>}
      ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x256xf32>)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul_with_duplicate_mmas_deduped
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = 1>
// CHECK:       iree_codegen.smt.constraints target = <set = 1>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK:       mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>]>

// -----

#gpu_target_block_mmas = #iree_gpu.target<arch = "gfx942", features = "", wgp = <
  compute = fp32, storage = b32, subgroup = shuffle,
  mma = [<MFMA_F32_4x4x4x16B_F16>, <MFMA_F32_16x16x16_F16>],
  subgroup_size_choices = [64],
  max_load_instruction_bits = 128,
  max_workgroup_sizes = [1024, 1024, 1024],
  max_thread_count_per_workgroup = 1024,
  max_workgroup_memory_bytes = 65536,
  max_workgroup_counts = [2147483647, 2147483647, 2147483647]
>>
#exec_target_block_mmas = #hal.executable.target<"rocm", "rocm-hsaco-fb",
    {iree_codegen.target_info = #gpu_target_block_mmas}>

func.func @matmul_with_block_intrinsic_filtered()
    attributes {hal.executable.target = #exec_target_block_mmas} {
  %cst = arith.constant 0.0 : f32
  %lhs = tensor.empty() : tensor<128x64xf16>
  %rhs = tensor.empty() : tensor<64x256xf16>
  %empty = tensor.empty() : tensor<128x256xf32>
  %fill = linalg.fill {root_op = #iree_codegen.root_op<set = 0>} ins(%cst : f32)
      outs(%empty : tensor<128x256xf32>) -> tensor<128x256xf32>
  %result = linalg.matmul {root_op = #iree_codegen.root_op<set = 0>}
      ins(%lhs, %rhs : tensor<128x64xf16>, tensor<64x256xf16>)
      outs(%fill : tensor<128x256xf32>) -> tensor<128x256xf32>
  return
}

// CHECK-LABEL: func.func @matmul_with_block_intrinsic_filtered
// CHECK:       linalg.matmul {{.+}} #iree_codegen.root_op<set = 0>
// CHECK:       iree_codegen.smt.constraints target = <set = 0>, pipeline = #iree_gpu.pipeline<VectorDistribute>,
// CHECK:       mma_kind = #iree_codegen.smt.one_of_knob<"mma_idx", [#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>]>
// CHECK-NOT:   MFMA_F32_4x4x4x16B_F16
