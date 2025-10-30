// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-tile-to-vector-size))" --split-input-file %s | FileCheck %s

#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 20, 0], vector_reduction = [0, 0, 30]>
func.func @matmul_all_dims_untiled(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @matmul_all_dims_untiled(
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               linalg.matmul

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 20, 0, 0], vector_reduction = [0, 0, 30, 30]>
func.func @invalid_matmul_vector_config(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @invalid_matmul_vector_config(
// CHECK-NOT:     scf.for
// CHECK:         linalg.matmul

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 30, 0], vector_reduction = [0, 0, 20]>
func.func @static_matmul_with_vector_size(%arg0 : tensor<10x20xf32>, %arg1 : tensor<20x30xf32>, %arg2 : tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20x30xf32>)
      outs(%arg2 : tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}
// CHECK-LABEL: func.func @static_matmul_with_vector_size(
// CHECK-NOT:     scf.for
// CHECK:         linalg.matmul

// -----

#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 30, 0], vector_reduction = [0, 0, 20]>
func.func @static_matmul_with_untiled_K_dim(%arg0 : tensor<10x40xf32>, %arg1 : tensor<40x30xf32>, %arg2 : tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul {lowering_config = #config}
      ins(%arg0, %arg1 : tensor<10x40xf32>, tensor<40x30xf32>)
      outs(%arg2 : tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}
// CHECK-LABEL: func.func @static_matmul_with_untiled_K_dim(
// CHECK:         %[[C20:.+]] = arith.constant 20 : index
// CHECK:         scf.for
// CHECK-SAME:      step %[[C20]]
// CHECK-NOT:     scf.for
// CHECK:           linalg.matmul

// -----

#map = affine_map<(d0)[s0] -> (-d0 + s0, 10)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 20)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 60)>
#config = #iree_cpu.lowering_config<vector_common_parallel = [10, 20, 0], vector_reduction = [0, 0, 30]>
func.func @matmul_tiled_MxNxK_to_10x20x60(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %N = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %K = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %mSize = affine.min #map(%c0)[%M]
  %nSize = affine.min #map1(%c0)[%N]
  %kSize = affine.min #map2(%c0)[%K]
  %lhs = tensor.extract_slice %arg0 [0, 0][%mSize, %kSize][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %rhs = tensor.extract_slice %arg1 [0, 0][%kSize, %nSize][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %acc = tensor.extract_slice %arg2 [0, 0][%mSize, %nSize][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %0 = linalg.matmul {lowering_config = #config}
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%acc : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @matmul_tiled_MxNxK_to_10x20x60(
// CHECK:         %[[C30:.+]] = arith.constant 30 : index
// CHECK:         scf.for
// CHECK-SAME:      step %[[C30]]
// CHECK-NOT:     scf.for
// CHECK:           linalg.matmul
