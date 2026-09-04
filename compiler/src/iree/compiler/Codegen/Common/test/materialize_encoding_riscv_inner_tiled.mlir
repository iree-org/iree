// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

// -----

// Materialization of data-tiled f16 matmul encodings to iree_codegen.inner_tiled
// on RISC-V V with Zvfh.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f16_zvl128b(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf16> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfh,+zvl128b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f16
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf16, #acc>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16, #acc>) -> tensor<?x?xf16, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf16, #lhs>, tensor<?x?xf16, #rhs>)
      outs(%3 : tensor<?x?xf16, #acc>) -> tensor<?x?xf16, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf16, #acc> -> tensor<?x?xf16>{%d0, %d1}
  return %5 : tensor<?x?xf16>
}
// CHECK-LABEL: func @matmul_f16_zvl128b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [14, 1]
//  CHECK-SAME:       -> tensor<?x?x14x1xf16>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK_LHS]] {{\[}}[0], [1], [2], [3, 4]{{\]}}
//  CHECK-SAME:       tensor<?x?x14x1xf16> into tensor<?x?x14x1x1xf16>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [16, 1]
//  CHECK-SAME:       -> tensor<?x?x16x1xf16>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[EXPANDED]], %[[PACK_RHS]])
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F16_F16, intrinsics_m = 14, vlen = 128>
//  CHECK-SAME:       tensor<?x?x14x1x1xf16>, tensor<?x?x16x1xf16> into tensor<?x?x14x1x16xf16>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INNER]] {{\[}}[0], [1], [2], [3, 4]{{\]}}
//  CHECK-SAME:       tensor<?x?x14x1x16xf16> into tensor<?x?x14x16xf16>
//       CHECK:   linalg.unpack %[[COLLAPSED]]

// -----

// The operand sizes scale with the target's vector length.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f16_zvl512b(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf16> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfh,+zvl512b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f16
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf16, #acc>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16, #acc>) -> tensor<?x?xf16, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf16, #lhs>, tensor<?x?xf16, #rhs>)
      outs(%3 : tensor<?x?xf16, #acc>) -> tensor<?x?xf16, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf16, #acc> -> tensor<?x?xf16>{%d0, %d1}
  return %5 : tensor<?x?xf16>
}
// CHECK-LABEL: func @matmul_f16_zvl512b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [14, 1]
//  CHECK-SAME:       -> tensor<?x?x14x1xf16>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK_LHS]] {{\[}}[0], [1], [2], [3, 4]{{\]}}
//  CHECK-SAME:       tensor<?x?x14x1xf16> into tensor<?x?x14x1x1xf16>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [64, 1]
//  CHECK-SAME:       -> tensor<?x?x64x1xf16>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[EXPANDED]], %[[PACK_RHS]])
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F16_F16, intrinsics_m = 14, vlen = 512>
//  CHECK-SAME:       tensor<?x?x14x1x1xf16>, tensor<?x?x64x1xf16> into tensor<?x?x14x1x64xf16>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INNER]] {{\[}}[0], [1], [2], [3, 4]{{\]}}
//  CHECK-SAME:       tensor<?x?x14x1x64xf16> into tensor<?x?x14x64xf16>
//       CHECK:   linalg.unpack %[[COLLAPSED]]

// -----

// Without Zvfh, selection falls back to the generic-scalar layout.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f16_no_zvfh(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf16> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f16
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf16, #acc>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<?x?xf16, #acc>) -> tensor<?x?xf16, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf16, #lhs>, tensor<?x?xf16, #rhs>)
      outs(%3 : tensor<?x?xf16, #acc>) -> tensor<?x?xf16, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf16, #acc> -> tensor<?x?xf16>{%d0, %d1}
  return %5 : tensor<?x?xf16>
}
// CHECK-LABEL: func @matmul_f16_no_zvfh(
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:       intrinsic = MMA_GENERIC_SCALAR_1x1x1_REG16
//   CHECK-NOT:       vlen

// -----

// f32 vfmacc.vf. N tile is vlen/8; intrinsics_m stays 6.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_zvl128b(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl128b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_zvl128b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [6, 1]
//  CHECK-SAME:       -> tensor<?x?x6x1xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK_LHS]] {{\[}}[0], [1], [2], [3, 4]{{\]}}
//  CHECK-SAME:       tensor<?x?x6x1xf32> into tensor<?x?x6x1x1xf32>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [16, 1]
//  CHECK-SAME:       -> tensor<?x?x16x1xf32>
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled ins(%[[EXPANDED]], %[[PACK_RHS]])
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 128>
//  CHECK-SAME:       tensor<?x?x6x1x1xf32>, tensor<?x?x16x1xf32> into tensor<?x?x6x1x16xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_zvl256b(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_zvl256b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [32, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 256>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_zvl512b(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl512b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_zvl512b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [64, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 512>

// -----

// Plain +v falls back to VLEN=128.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f32_plain_v(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #lhs>, tensor<?x?xf32, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f32_plain_v(
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 128>

// -----

// f16→f32 CASTF32 via Zvfhmin. N tile is vlen/8; cost model picks intrinsics_m = 7.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f16_castf32_zvl128b(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfhmin,+zvl128b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf16, #lhs>, tensor<?x?xf16, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f16_castf32_zvl128b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [7, 1]
//  CHECK-SAME:       -> tensor<?x?x7x1xf16>
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [16, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F16_CASTF32, intrinsics_m = 7, vlen = 128>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f16_castf32_zvl256b(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfhmin,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf16, #lhs>, tensor<?x?xf16, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f16_castf32_zvl256b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [32, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F16_CASTF32, intrinsics_m = 7, vlen = 256>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f16_castf32_zvl512b(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfhmin,+zvl512b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf16, #lhs>, tensor<?x?xf16, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f16_castf32_zvl512b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [64, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F16_CASTF32, intrinsics_m = 7, vlen = 512>

// -----

// +zvfh implies +zvfhmin, so f16f16f32 still selects CASTF32.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_f16_castf32_zvfh_implies_zvfhmin(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?xf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfh,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xf16> -> tensor<?x?xf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf16, #lhs>, tensor<?x?xf16, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_f16_castf32_zvfh_implies_zvfhmin(
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F16_CASTF32, intrinsics_m = 7, vlen = 256>

// -----

// bf16→f32 vfwmaccbf16. N tile is vlen/8; cost model picks intrinsics_m = 7.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_bf16_zvl128b(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfbfwma,+zvl128b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xbf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xbf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xbf16, #lhs>, tensor<?x?xbf16, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_bf16_zvl128b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [7, 1]
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [16, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFWMACCBF16_1x8VLsx1_F32_BF16, intrinsics_m = 7, vlen = 128>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_bf16_zvl256b(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfbfwma,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xbf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xbf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xbf16, #lhs>, tensor<?x?xbf16, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_bf16_zvl256b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [32, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFWMACCBF16_1x8VLsx1_F32_BF16, intrinsics_m = 7, vlen = 256>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_bf16_zvl512b(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvfbfwma,+zvl512b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xbf16>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xbf16>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xbf16> -> tensor<?x?xbf16, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #acc>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xbf16, #lhs>, tensor<?x?xbf16, #rhs>)
      outs(%3 : tensor<?x?xf32, #acc>) -> tensor<?x?xf32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xf32, #acc> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
// CHECK-LABEL: func @matmul_bf16_zvl512b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [64, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFWMACCBF16_1x8VLsx1_F32_BF16, intrinsics_m = 7, vlen = 512>

// -----

// i8→i32 CASTI16. N tile is vlen/8; cost model picks intrinsics_m = 7.
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_i8_zvl128b(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?xi8>, %m: index, %n: index, %k: index) -> tensor<?x?xi32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl128b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0 : i32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xi8>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xi8> -> tensor<?x?xi8, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xi8> -> tensor<?x?xi8, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xi32, #acc>
  %3 = linalg.fill ins(%cst : i32) outs(%2 : tensor<?x?xi32, #acc>) -> tensor<?x?xi32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xi8, #lhs>, tensor<?x?xi8, #rhs>)
      outs(%3 : tensor<?x?xi32, #acc>) -> tensor<?x?xi32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xi32, #acc> -> tensor<?x?xi32>{%d0, %d1}
  return %5 : tensor<?x?xi32>
}
// CHECK-LABEL: func @matmul_i8_zvl128b(
//       CHECK:   %[[PACK_LHS:.+]] = linalg.pack {{.*}}inner_tiles = [7, 1]
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [16, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VWMACC_1x8VLsx1_I32_I8_CASTI16, intrinsics_m = 7, vlen = 128>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_i8_zvl256b(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?xi8>, %m: index, %n: index, %k: index) -> tensor<?x?xi32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0 : i32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xi8>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xi8> -> tensor<?x?xi8, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xi8> -> tensor<?x?xi8, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xi32, #acc>
  %3 = linalg.fill ins(%cst : i32) outs(%2 : tensor<?x?xi32, #acc>) -> tensor<?x?xi32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xi8, #lhs>, tensor<?x?xi8, #rhs>)
      outs(%3 : tensor<?x?xi32, #acc>) -> tensor<?x?xi32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xi32, #acc> -> tensor<?x?xi32>{%d0, %d1}
  return %5 : tensor<?x?xi32>
}
// CHECK-LABEL: func @matmul_i8_zvl256b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [32, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VWMACC_1x8VLsx1_I32_I8_CASTI16, intrinsics_m = 7, vlen = 256>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#acc = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_i8_zvl512b(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?xi8>, %m: index, %n: index, %k: index) -> tensor<?x?xi32> attributes {
   hal.executable.target = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {target_triple = "riscv64-unknown-unknown-eabi-elf", cpu_features = "+m,+a,+f,+d,+c,+v,+zvl512b", enable_inner_tiled = true, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0 : i32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xi8>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xi8>
  %0 = iree_encoding.set_encoding %arg0 encoding_dims{%m, %n, %k} : tensor<?x?xi8> -> tensor<?x?xi8, #lhs>
  %1 = iree_encoding.set_encoding %arg1 encoding_dims{%m, %n, %k} : tensor<?x?xi8> -> tensor<?x?xi8, #rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xi32, #acc>
  %3 = linalg.fill ins(%cst : i32) outs(%2 : tensor<?x?xi32, #acc>) -> tensor<?x?xi32, #acc>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xi8, #lhs>, tensor<?x?xi8, #rhs>)
      outs(%3 : tensor<?x?xi32, #acc>) -> tensor<?x?xi32, #acc>
  %5 = iree_encoding.unset_encoding %4 encoding_dims{%m, %n, %k} : tensor<?x?xi32, #acc> -> tensor<?x?xi32>{%d0, %d1}
  return %5 : tensor<?x?xi32>
}
// CHECK-LABEL: func @matmul_i8_zvl512b(
//       CHECK:   %[[PACK_RHS:.+]] = linalg.pack {{.*}}inner_tiles = [64, 1]
//       CHECK:   %[[INNER:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:       kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VWMACC_1x8VLsx1_I32_I8_CASTI16, intrinsics_m = 7, vlen = 512>
