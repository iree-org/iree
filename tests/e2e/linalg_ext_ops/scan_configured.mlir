#translation = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>
#config = #iree_gpu.lowering_config<{
    lane_basis = [[8], [0]],
    subgroup_basis = [[1], [0]],
    thread = [8]
}>
#compilation = #iree_codegen.compilation_info<
    lowering_config = #config,
    translation_info = #translation>

func.func @scan_dim0_inclusive_sum_configured_rocm() {
  %c1 = arith.constant 1 : index
  %input_empty = tensor.empty() : tensor<256xf32>
  %input_init = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%input_empty : tensor<256xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx1 = arith.addi %idx, %c1 : index
      %idx_i32 = arith.index_cast %idx1 : index to i32
      %value = arith.sitofp %idx_i32 : i32 to f32
      linalg.yield %value : f32
  } -> tensor<256xf32>
  %input = util.optimization_barrier %input_init : tensor<256xf32>

  %output = tensor.empty() : tensor<256xf32>
  %accum = util.unfoldable_constant dense<0.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan {compilation_info = #compilation}
         dimension(0) inclusive(true)
         ins(%input : tensor<256xf32>)
         outs(%output, %accum : tensor<256xf32>, tensor<f32>) {
           ^bb0(%lhs : f32, %rhs : f32):
             %sum = arith.addf %lhs, %rhs : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<256xf32>, tensor<f32>

  %prefix = tensor.extract_slice %0#0[0] [4] [1]
      : tensor<256xf32> to tensor<4xf32>
  check.expect_almost_eq_const(
      %prefix,
      dense<[1.0, 3.0, 6.0, 10.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<32896.0> : tensor<f32>
  ) : tensor<f32>

  return
}

#translation_exclusive = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>
#config_exclusive = #iree_gpu.lowering_config<{
    lane_basis = [[16], [0]],
    subgroup_basis = [[1], [0]],
    thread = [16]
}>
#compilation_exclusive = #iree_codegen.compilation_info<
    lowering_config = #config_exclusive,
    translation_info = #translation_exclusive>

func.func @scan_dim0_exclusive_sum_configured_rocm() {
  %c1 = arith.constant 1 : index
  %input_empty = tensor.empty() : tensor<256xf32>
  %input_init = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%input_empty : tensor<256xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx1 = arith.addi %idx, %c1 : index
      %idx_i32 = arith.index_cast %idx1 : index to i32
      %value = arith.sitofp %idx_i32 : i32 to f32
      linalg.yield %value : f32
  } -> tensor<256xf32>
  %input = util.optimization_barrier %input_init : tensor<256xf32>

  %output = tensor.empty() : tensor<256xf32>
  %accum = util.unfoldable_constant dense<5.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan {compilation_info = #compilation_exclusive}
         dimension(0) inclusive(false)
         ins(%input : tensor<256xf32>)
         outs(%output, %accum : tensor<256xf32>, tensor<f32>) {
           ^bb0(%lhs : f32, %rhs : f32):
             %sum = arith.addf %lhs, %rhs : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<256xf32>, tensor<f32>

  %prefix = tensor.extract_slice %0#0[0] [4] [1]
      : tensor<256xf32> to tensor<4xf32>
  check.expect_almost_eq_const(
      %prefix,
      dense<[5.0, 6.0, 8.0, 11.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  %boundary = tensor.extract_slice %0#0[14] [4] [1]
      : tensor<256xf32> to tensor<4xf32>
  check.expect_almost_eq_const(
      %boundary,
      dense<[110.0, 125.0, 141.0, 158.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<32645.0> : tensor<f32>
  ) : tensor<f32>

  return
}

#translation_large = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [64, 1, 1] subgroup_size = 64>
#config_large = #iree_gpu.lowering_config<{
    lane_basis = [[2, 8], [0, 1]],
    subgroup_basis = [[1, 1], [0, 1]],
    thread = [2, 8]
}>
#compilation_large = #iree_codegen.compilation_info<
    lowering_config = #config_large,
    translation_info = #translation_large>

func.func @scan_dim1_inclusive_sum_large_configured_rocm() {
  %c1 = arith.constant 1 : index
  %c1000 = arith.constant 1000 : index
  %input_empty = tensor.empty() : tensor<64x256xf32>
  %input_init = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%input_empty : tensor<64x256xf32>) {
    ^bb0(%out: f32):
      %row = linalg.index 0 : index
      %col = linalg.index 1 : index
      %row_base = arith.muli %row, %c1000 : index
      %col1 = arith.addi %col, %c1 : index
      %idx = arith.addi %row_base, %col1 : index
      %idx_i32 = arith.index_cast %idx : index to i32
      %value = arith.sitofp %idx_i32 : i32 to f32
      linalg.yield %value : f32
  } -> tensor<64x256xf32>
  %input = util.optimization_barrier %input_init : tensor<64x256xf32>

  %output = tensor.empty() : tensor<64x256xf32>
  %accum = util.unfoldable_constant dense<0.0> : tensor<64xf32>
  %0:2 = iree_linalg_ext.scan {compilation_info = #compilation_large}
         dimension(1) inclusive(true)
         ins(%input : tensor<64x256xf32>)
         outs(%output, %accum : tensor<64x256xf32>, tensor<64xf32>) {
           ^bb0(%lhs : f32, %rhs : f32):
             %sum = arith.addf %lhs, %rhs : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<64x256xf32>, tensor<64xf32>

  %prefix = tensor.extract_slice %0#0[0, 0] [2, 4] [1, 1]
      : tensor<64x256xf32> to tensor<2x4xf32>
  check.expect_almost_eq_const(
      %prefix,
      dense<[[1.0, 3.0, 6.0, 10.0],
             [1001.0, 2003.0, 3006.0, 4010.0]]> : tensor<2x4xf32>
  ) : tensor<2x4xf32>

  %boundary = tensor.extract_slice %0#0[0, 14] [2, 4] [1, 1]
      : tensor<64x256xf32> to tensor<2x4xf32>
  check.expect_almost_eq_const(
      %boundary,
      dense<[[120.0, 136.0, 153.0, 171.0],
             [15120.0, 16136.0, 17153.0, 18171.0]]> : tensor<2x4xf32>
  ) : tensor<2x4xf32>

  %acc_prefix = tensor.extract_slice %0#1[0] [2] [1]
      : tensor<64xf32> to tensor<2xf32>
  check.expect_almost_eq_const(
      %acc_prefix,
      dense<[32896.0, 288896.0]> : tensor<2xf32>
  ) : tensor<2xf32>

  return
}

#translation_cross_subgroup = #iree_codegen.translation_info<
    pipeline = #iree_gpu.pipeline<VectorDistribute> workgroup_size = [128, 1, 1] subgroup_size = 64>
#config_cross_subgroup = #iree_gpu.lowering_config<{
    subgroup_basis = [[2], [0]],
    lane_basis = [[8], [0]],
    thread = [4]
}>
#compilation_cross_subgroup = #iree_codegen.compilation_info<
    lowering_config = #config_cross_subgroup,
    translation_info = #translation_cross_subgroup>

func.func @scan_dim0_inclusive_sum_cross_subgroup_configured_rocm() {
  %c1 = arith.constant 1 : index
  %input_empty = tensor.empty() : tensor<128xf32>
  %input_init = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%input_empty : tensor<128xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx1 = arith.addi %idx, %c1 : index
      %idx_i32 = arith.index_cast %idx1 : index to i32
      %value = arith.sitofp %idx_i32 : i32 to f32
      linalg.yield %value : f32
  } -> tensor<128xf32>
  %input = util.optimization_barrier %input_init : tensor<128xf32>

  %output = tensor.empty() : tensor<128xf32>
  %accum = util.unfoldable_constant dense<0.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan {compilation_info = #compilation_cross_subgroup}
         dimension(0) inclusive(true)
         ins(%input : tensor<128xf32>)
         outs(%output, %accum : tensor<128xf32>, tensor<f32>) {
           ^bb0(%lhs : f32, %rhs : f32):
             %sum = arith.addf %lhs, %rhs : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<128xf32>, tensor<f32>

  %prefix = tensor.extract_slice %0#0[0] [4] [1]
      : tensor<128xf32> to tensor<4xf32>
  check.expect_almost_eq_const(
      %prefix,
      dense<[1.0, 3.0, 6.0, 10.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  %boundary = tensor.extract_slice %0#0[62] [4] [1]
      : tensor<128xf32> to tensor<4xf32>
  check.expect_almost_eq_const(
      %boundary,
      dense<[2016.0, 2080.0, 2145.0, 2211.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<8256.0> : tensor<f32>
  ) : tensor<f32>

  return
}

func.func @scan_dim0_exclusive_sum_cross_subgroup_configured_rocm() {
  %c1 = arith.constant 1 : index
  %input_empty = tensor.empty() : tensor<128xf32>
  %input_init = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%input_empty : tensor<128xf32>) {
    ^bb0(%out: f32):
      %idx = linalg.index 0 : index
      %idx1 = arith.addi %idx, %c1 : index
      %idx_i32 = arith.index_cast %idx1 : index to i32
      %value = arith.sitofp %idx_i32 : i32 to f32
      linalg.yield %value : f32
  } -> tensor<128xf32>
  %input = util.optimization_barrier %input_init : tensor<128xf32>

  %output = tensor.empty() : tensor<128xf32>
  %accum = util.unfoldable_constant dense<5.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan {compilation_info = #compilation_cross_subgroup}
         dimension(0) inclusive(false)
         ins(%input : tensor<128xf32>)
         outs(%output, %accum : tensor<128xf32>, tensor<f32>) {
           ^bb0(%lhs : f32, %rhs : f32):
             %sum = arith.addf %lhs, %rhs : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<128xf32>, tensor<f32>

  %prefix = tensor.extract_slice %0#0[0] [4] [1]
      : tensor<128xf32> to tensor<4xf32>
  check.expect_almost_eq_const(
      %prefix,
      dense<[5.0, 6.0, 8.0, 11.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  %boundary = tensor.extract_slice %0#0[62] [4] [1]
      : tensor<128xf32> to tensor<4xf32>
  check.expect_almost_eq_const(
      %boundary,
      dense<[1958.0, 2021.0, 2085.0, 2150.0]> : tensor<4xf32>
  ) : tensor<4xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<8133.0> : tensor<f32>
  ) : tensor<f32>

  return
}
