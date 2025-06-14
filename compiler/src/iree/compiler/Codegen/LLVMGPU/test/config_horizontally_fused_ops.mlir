// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK,GFX942
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' --mlir-print-local-scope %s | FileCheck %s --check-prefixes=CHECK,GFX950

func.func @fused_contraction_1(%arg0: tensor<2x4096x640xf16>,
    %arg1 : tensor<10x64x640xf16>, %arg2 : tensor<10x64x640xf16>,
    %arg3 : tensor<10x64x640xf16>)
    -> (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>) {
  %11 = tensor.empty() : tensor<2x10x4096x64xf16>
  %12 = tensor.empty() : tensor<2x10x4096x64xf32>
  %cst = arith.constant 0.0: f32
  %13 = linalg.fill ins(%cst : f32)
      outs(%12 : tensor<2x10x4096x64xf32>) -> tensor<2x10x4096x64xf32>
  %14:3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
          : tensor<2x4096x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>,
            tensor<10x64x640xf16>)
      outs(%13, %13, %13
          : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f32, %out_3: f32, %out_4: f32):
      %18 = arith.extf %in : f16 to f32
      %19 = arith.extf %in_0 : f16 to f32
      %20 = arith.mulf %18, %19 : f32
      %21 = arith.addf %out, %20 : f32
      %22 = arith.extf %in_1 : f16 to f32
      %23 = arith.mulf %18, %22 : f32
      %24 = arith.addf %out_3, %23 : f32
      %25 = arith.extf %in_2 : f16 to f32
      %26 = arith.mulf %18, %25 : f32
      %27 = arith.addf %out_4, %26 : f32
      linalg.yield %21, %24, %27 : f32, f32, f32
  } -> (tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>)
  %15 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14#0 : tensor<2x10x4096x64xf32>) outs(%11 : tensor<2x10x4096x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %18 = arith.truncf %in : f32 to f16
      linalg.yield %18 : f16
  } -> tensor<2x10x4096x64xf16>
  %16 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14#1 : tensor<2x10x4096x64xf32>) outs(%11 : tensor<2x10x4096x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %18 = arith.truncf %in : f32 to f16
      linalg.yield %18 : f16
    } -> tensor<2x10x4096x64xf16>
  %17 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14#2 : tensor<2x10x4096x64xf32>) outs(%11 : tensor<2x10x4096x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %18 = arith.truncf %in : f32 to f16
      linalg.yield %18 : f16
  } -> tensor<2x10x4096x64xf16>
  return %15, %16, %17
      : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>
}
// CHECK-LABEL: func @fused_contraction_1
//  CHECK-SAME:     translation_info = #iree_codegen.translation_info
//  CHECK-SAME:         pipeline = LLVMGPUVectorDistribute
//  CHECK-SAME:         workgroup_size = [256, 1, 1]
//  CHECK-SAME:         subgroup_size = 64
//       CHECK:   %[[GENERIC:.+]]:3 = linalg.generic
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config
// GFX942-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16
// GFX950-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16
//  CHECK-SAME:       promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:       reduction = [0, 0, 0, 0, 128]
//  CHECK-SAME:       subgroup_m_count = 2
//  CHECK-SAME:       subgroup_n_count = 2
//  CHECK-SAME:       workgroup = [1, 1, 32, 32, 0]

// -----

func.func @fused_contraction_2(%arg0: tensor<4096x640xf32>,
    %arg1 : tensor<640x640xf32>, %arg2 : tensor<640x640xf32>,
    %arg3 : tensor<640x640xf32>)
    -> (tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>) {
  %11 = tensor.empty() : tensor<4096x640xf32>
  %12 = tensor.empty() : tensor<4096x640xf32>
  %cst = arith.constant 0.0: f32
  %13 = linalg.fill ins(%cst : f32)
      outs(%12 : tensor<4096x640xf32>) -> tensor<4096x640xf32>
  %14:3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
          : tensor<4096x640xf32>, tensor<640x640xf32>, tensor<640x640xf32>,
            tensor<640x640xf32>)
      outs(%13, %13, %13
          : tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %in_2: f32, %out: f32, %out_3: f32, %out_4: f32):
      %20 = arith.mulf %in, %in_0 : f32
      %21 = arith.addf %out, %20 : f32
      %23 = arith.mulf %in, %in_1 : f32
      %24 = arith.addf %out_3, %23 : f32
      %26 = arith.mulf %in, %in_2 : f32
      %27 = arith.addf %out_4, %26 : f32
      linalg.yield %21, %24, %27 : f32, f32, f32
  } -> (tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>)
  return %14#0, %14#1, %14#2
      : tensor<4096x640xf32>, tensor<4096x640xf32>, tensor<4096x640xf32>
}
// CHECK-LABEL: func @fused_contraction_2
//  CHECK-SAME:     translation_info = #iree_codegen.translation_info
//  CHECK-SAME:         pipeline = LLVMGPUVectorDistribute
//  CHECK-SAME:         workgroup_size = [256, 1, 1]
//  CHECK-SAME:         subgroup_size = 64
//       CHECK:   %[[GENERIC:.+]]:3 = linalg.generic
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config
// GFX942-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32
// GFX950-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32
//  CHECK-SAME:       promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:       reduction = [0, 0, 16]
//  CHECK-SAME:       subgroup_m_count = 2
//  CHECK-SAME:       subgroup_n_count = 2
//  CHECK-SAME:       workgroup = [32, 64, 0]

// -----

func.func @fused_contraction_3(%arg0 : tensor<2x4096x640xi8>,
    %arg1 : tensor<2x640x640xi8>, %arg2 : tensor<2x640x640xi8>)
    -> (tensor<2x4096x640xf16>, tensor<2x4096x640xf16>) {
  %c0_i32 = arith.constant 0 : i32
  %18 = tensor.empty() : tensor<2x4096x640xf16>
  %19 = tensor.empty() : tensor<2x4096x640xi32>
  %20 = linalg.fill ins(%c0_i32 : i32)
      outs(%19 : tensor<2x4096x640xi32>) -> tensor<2x4096x640xi32>
  %21:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>, tensor<2x640x640xi8>)
      outs(%20, %20 : tensor<2x4096x640xi32>, tensor<2x4096x640xi32>) {
    ^bb0(%in: i8, %in_0: i8, %in_1: i8, %out: i32, %out_2: i32):
      %24 = arith.extsi %in : i8 to i32
      %25 = arith.extsi %in_0 : i8 to i32
      %26 = arith.muli %24, %25 : i32
      %27 = arith.addi %out, %26 : i32
      %28 = arith.extsi %in_1 : i8 to i32
      %29 = arith.muli %24, %28 : i32
      %30 = arith.addi %out_2, %29 : i32
      linalg.yield %27, %30 : i32, i32
  } -> (tensor<2x4096x640xi32>, tensor<2x4096x640xi32>)
  %22 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%21#0 : tensor<2x4096x640xi32>) outs(%18 : tensor<2x4096x640xf16>) {
    ^bb0(%in: i32, %out: f16):
      %27 = arith.sitofp %in : i32 to f32
      %29 = arith.truncf %27 : f32 to f16
      linalg.yield %29 : f16
  } -> tensor<2x4096x640xf16>
  %23 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%21#1 : tensor<2x4096x640xi32>) outs(%18 : tensor<2x4096x640xf16>) {
    ^bb0(%in: i32, %out: f16):
      %27 = arith.sitofp %in : i32 to f32
      %29 = arith.truncf %27 : f32 to f16
      linalg.yield %29 : f16
  } -> tensor<2x4096x640xf16>
  return %22, %23 : tensor<2x4096x640xf16>, tensor<2x4096x640xf16>
}
// CHECK-LABEL: func @fused_contraction_3
//  CHECK-SAME:     translation_info = #iree_codegen.translation_info
//  CHECK-SAME:         pipeline = LLVMGPUVectorDistribute
//  CHECK-SAME:         workgroup_size = [256, 1, 1]
//  CHECK-SAME:         subgroup_size = 64
//       CHECK:   %[[GENERIC:.+]]:2 = linalg.generic
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config
// GFX942-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8
// GFX950-SAME:      mma_kind = #iree_gpu.mma_layout<MFMA_I32_16x16x64_I8
//  CHECK-SAME:       promote_operands = [0, 1, 2]
//  CHECK-SAME:       reduction = [0, 0, 0, 128]
//  CHECK-SAME:       subgroup_m_count = 2
//  CHECK-SAME:       subgroup_n_count = 2
//  CHECK-SAME:       workgroup = [1, 64, 64, 0]

// -----

func.func @fused_contraction_4(%arg0: tensor<2x4096x640xf16>,
    %arg1 : tensor<10x64x640xf16>, %arg2 : tensor<10x64x640xf16>,
    %arg3 : tensor<10x64x640xf16>)
    -> (tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x64x4096xf16>) {
  %9 = tensor.empty() : tensor<2x10x64x4096xf16>
  %10 = tensor.empty() : tensor<2x10x64x4096xf32>
  %11 = tensor.empty() : tensor<2x10x4096x64xf16>
  %12 = tensor.empty() : tensor<2x10x4096x64xf32>
  %cst = arith.constant 0.0: f32
  %fill0 = linalg.fill ins(%cst : f32)
      outs(%12 : tensor<2x10x4096x64xf32>) -> tensor<2x10x4096x64xf32>
  %fill1 = linalg.fill ins(%cst : f32)
      outs(%10 : tensor<2x10x64x4096xf32>) -> tensor<2x10x64x4096xf32>
  %14:3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
          : tensor<2x4096x640xf16>, tensor<10x64x640xf16>, tensor<10x64x640xf16>,
            tensor<10x64x640xf16>)
      outs(%fill0, %fill0, %fill1
          : tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x64x4096xf32>) {
    ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f32, %out_3: f32, %out_4: f32):
      %18 = arith.extf %in : f16 to f32
      %19 = arith.extf %in_0 : f16 to f32
      %20 = arith.mulf %18, %19 : f32
      %21 = arith.addf %out, %20 : f32
      %22 = arith.extf %in_1 : f16 to f32
      %23 = arith.mulf %18, %22 : f32
      %24 = arith.addf %out_3, %23 : f32
      %25 = arith.extf %in_2 : f16 to f32
      %26 = arith.mulf %18, %25 : f32
      %27 = arith.addf %out_4, %26 : f32
      linalg.yield %21, %24, %27 : f32, f32, f32
  } -> (tensor<2x10x4096x64xf32>, tensor<2x10x4096x64xf32>, tensor<2x10x64x4096xf32>)
  %15 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14#0 : tensor<2x10x4096x64xf32>) outs(%11 : tensor<2x10x4096x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %18 = arith.truncf %in : f32 to f16
      linalg.yield %18 : f16
  } -> tensor<2x10x4096x64xf16>
  %16 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14#1 : tensor<2x10x4096x64xf32>) outs(%11 : tensor<2x10x4096x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %18 = arith.truncf %in : f32 to f16
      linalg.yield %18 : f16
    } -> tensor<2x10x4096x64xf16>
  %17 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%14#2 : tensor<2x10x64x4096xf32>) outs(%9 : tensor<2x10x64x4096xf16>) {
    ^bb0(%in: f32, %out: f16):
      %18 = arith.truncf %in : f32 to f16
      linalg.yield %18 : f16
  } -> tensor<2x10x64x4096xf16>
  return %15, %16, %17
      : tensor<2x10x4096x64xf16>, tensor<2x10x4096x64xf16>, tensor<2x10x64x4096xf16>
}
// CHECK-LABEL: func @fused_contraction_4
//  CHECK-SAME:     translation_info = #iree_codegen.translation_info
//  CHECK-SAME:         pipeline = LLVMGPUVectorDistribute
//  CHECK-SAME:         workgroup_size = [256, 1, 1]
//  CHECK-SAME:         subgroup_size = 64
//       CHECK:   %[[GENERIC:.+]]:3 = linalg.generic
//  CHECK-SAME:       lowering_config = #iree_gpu.lowering_config
// GFX942-SAME:       mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16
// GFX950-SAME:       mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16
//  CHECK-SAME:       promote_operands = [0, 1, 2, 3]
//  CHECK-SAME:       reduction = [0, 0, 0, 0, 128]
//  CHECK-SAME:       subgroup_m_count = 2
//  CHECK-SAME:       subgroup_n_count = 2
//  CHECK-SAME:       workgroup = [1, 1, 32, 32, 0]
