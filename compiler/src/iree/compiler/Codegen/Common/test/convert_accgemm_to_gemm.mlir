// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-convert-accgemm-to-gemm))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @accumulate_gemm(%1 : tensor<512x128xi8>, %2 : tensor<512x128xi8>) {
  %c0 = arith.constant 0 : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x512xi32>>
  %4 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x512xi32>> -> tensor<512x512xi32>
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%1, %2 : tensor<512x128xi8>, tensor<512x128xi8>) outs(%4 : tensor<512x512xi32>) {
      ^bb0(%in: i8, %in_0: i8, %out: i32):
        %6 = arith.extsi %in : i8 to i32
        %7 = arith.extsi %in_0 : i8 to i32
        %8 = arith.muli %6, %7 : i32
        %9 = arith.addi %out, %8 : i32
        linalg.yield %9 : i32
      } -> tensor<512x512xi32>
  iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : tensor<512x512xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x512xi32>>
  return
}

// CHECK-LABEL: func.func @accumulate_gemm
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<512x512xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<512x512xi32>) -> tensor<512x512xi32>
//       CHECK: %[[GEMM:.+]] = linalg.generic {{.*}} outs(%[[FILL]] : tensor<512x512xi32>) {
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[GEMM]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: iree_tensor_ext.dispatch.tensor.store %[[ADD]]

// -----

#lhs_map = affine_map<(M, N, Ko, Kb) -> (M, Ko, Kb)>
#rhs_map = affine_map<(M, N, Ko, Kb) -> (N, Ko, Kb)>
#scale_m = affine_map<(M, N, Ko, Kb) -> (M, Ko)>
#scale_n = affine_map<(M, N, Ko, Kb) -> (N, Ko)>
#out_map = affine_map<(M, N, Ko, Kb) -> (M, N)>
func.func @accumulate_scaled_gemm(
    %A: tensor<1024x512x32xf4E2M1FN>, %B: tensor<1024x512x32xf4E2M1FN>,
    %A_scales: tensor<1024x512xf8E8M0FNU>, %B_scales: tensor<1024x512xf8E8M0FNU>,
    %C_buffer: memref<1024x1024xf32>) {
  %C = iree_codegen.load_from_buffer %C_buffer : memref<1024x1024xf32> -> tensor<1024x1024xf32>
  %0 = linalg.generic {
    indexing_maps = [#lhs_map, #rhs_map, #scale_m, #scale_n, #out_map],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]
  } ins(%A, %B, %A_scales, %B_scales : tensor<1024x512x32xf4E2M1FN>, tensor<1024x512x32xf4E2M1FN>, tensor<1024x512xf8E8M0FNU>, tensor<1024x512xf8E8M0FNU>) outs(%C : tensor<1024x1024xf32>) {
  ^bb0(%a: f4E2M1FN, %b: f4E2M1FN, %a_scale: f8E8M0FNU, %b_scale: f8E8M0FNU, %out: f32):
    %1 = arith.scaling_extf %a, %a_scale : f4E2M1FN, f8E8M0FNU to f32
    %2 = arith.scaling_extf %b, %b_scale : f4E2M1FN, f8E8M0FNU to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<1024x1024xf32>
  iree_codegen.store_to_buffer %0, %C_buffer : tensor<1024x1024xf32> into memref<1024x1024xf32>
  return
}

// CHECK-LABEL: func.func @accumulate_scaled_gemm
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<1024x1024xf32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
//       CHECK: %[[GEMM:.+]] = linalg.generic {{.*}} outs(%[[FILL]] : tensor<1024x1024xf32>) {
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[GEMM]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: iree_codegen.store_to_buffer %[[ADD]]

// -----
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @accumulate_inner_tiled(%1 : tensor<?x?x4xf16>, %2 : tensor<?x?x4xf16>, %3 : memref<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %4 = iree_codegen.load_from_buffer %3 : memref<?x?x4xf32> -> tensor<?x?x4xf32>
  %5 = iree_codegen.inner_tiled ins(%1, %2) outs(%4) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>,
                     affine_map<(i, j, k) -> (k, j)>,
                     affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %5 : tensor<?x?x4xf32>
}

// CHECK-LABEL: func.func @accumulate_inner_tiled
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<?x?x4xf32>) -> tensor<?x?x4xf32>
//       CHECK: %[[GEMM:.+]] = iree_codegen.inner_tiled {{.*}} outs(%[[FILL]])
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[GEMM]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: return %[[ADD]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @acc_conv_nchw(%1 : tensor<1x64x58x58xf32>, %2 : tensor<64x64x3x3xf32>) {
  %c0 = arith.constant 0 : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x64x56x56xf32>>
  %4 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0, 0], sizes = [1, 64, 56, 56], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x64x56x56xf32>> -> tensor<1x64x56x56xf32>
  %5 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%1, %2 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%4 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0, 0, 0], sizes = [1, 64, 56, 56], strides = [1, 1, 1, 1] : tensor<1x64x56x56xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x64x56x56xf32>>
  return
}

// CHECK-LABEL: func.func @acc_conv_nchw
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<1x64x56x56xf32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
//       CHECK: %[[CONV:.+]] = linalg.conv_2d_nchw_fchw {{.*}} outs(%[[FILL]] : tensor<1x64x56x56xf32>)
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[CONV]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: iree_tensor_ext.dispatch.tensor.store %[[ADD]]

// -----
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @nonacc_gemm(%1 : tensor<512x128xi8>, %2 : tensor<512x128xi8>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x512xi32>>
  %empty = tensor.empty() : tensor<512x512xi32>
  %fill = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<512x512xi32>) -> tensor<512x512xi32>
  %5 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%1, %2 : tensor<512x128xi8>, tensor<512x128xi8>) outs(%fill : tensor<512x512xi32>) -> tensor<512x512xi32>
  iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : tensor<512x512xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x512xi32>>
  return
}

// CHECK-DAG: #[[$MA:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MB:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[$MC:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @nonacc_gemm
//       CHECK: linalg.matmul indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]]
//   CHECK-NOT: linalg.generic
