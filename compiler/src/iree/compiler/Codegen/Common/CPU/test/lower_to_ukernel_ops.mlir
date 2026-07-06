// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-cpu-lower-to-ukernels{skip-intermediate-roundings=true},cse,canonicalize))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-cpu-lower-to-ukernels{skip-intermediate-roundings=false},cse,canonicalize))" %s | FileCheck %s --check-prefix=NOSKIPROUND

func.func @mmt4d_f32f32f32(%arg0 : tensor<?x?x16x1xf32>, %arg1 : tensor<?x?x16x1xf32>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
      outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_f32f32f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C1_i32:.+]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C16_i32]], %[[C16_i32]], %[[C1_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

func.func @mmt4d_no_ukernels_attr_f32f32f32(%arg0 : tensor<?x?x16x1xf32>, %arg1 : tensor<?x?x16x1xf32>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
      outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_no_ukernels_attr_f32f32f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C1_i32:.+]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C16_i32]], %[[C16_i32]], %[[C1_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

func.func @mmt4d_f32f32f32_with_none_ukernel_enabled(%arg0 : tensor<?x?x16x1xf32>, %arg1 : tensor<?x?x16x1xf32>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "none", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
      outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_f32f32f32_with_none_ukernel_enabled(
//      CHECK:   linalg.mmt4d

// -----

func.func @mmt4d_f32f32f32_with_only_mmt4d_ukernel_enabled(%arg0 : tensor<?x?x16x1xf32>, %arg1 : tensor<?x?x16x1xf32>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "mmt4d", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
      outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_f32f32f32_with_only_mmt4d_ukernel_enabled(
//      CHECK:   iree_codegen.ukernel.generic "iree_uk_mmt4d"

// -----

func.func @mmt4d_f32f32f32_with_only_foo_mmt4d_bar_ukernel_enabled(%arg0 : tensor<?x?x16x1xf32>, %arg1 : tensor<?x?x16x1xf32>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "foo,mmt4d,bar", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
      outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_f32f32f32_with_only_foo_mmt4d_bar_ukernel_enabled(
//      CHECK:   iree_codegen.ukernel.generic "iree_uk_mmt4d"

// -----

func.func @mmt4d_f32f32f32_with_only_foo_ukernel_enabled(%arg0 : tensor<?x?x16x1xf32>, %arg1 : tensor<?x?x16x1xf32>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "foo", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
      outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_f32f32f32_with_only_foo_ukernel_enabled(
//      CHECK:   linalg.mmt4d

// -----

func.func @mmt4d_fill(%arg0 : tensor<?x?x16x1xf32>, %arg1 : tensor<?x?x16x1xf32>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf32>, tensor<?x?x16x1xf32>)
      outs(%fill : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_fill(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C1_i32:.+]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C16_i32]], %[[C16_i32]], %[[C1_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0


// -----

func.func @mmt4d_i8i8i32(%arg0 : tensor<?x?x16x2xi8>, %arg1 : tensor<?x?x16x2xi8>,
    %arg2 : tensor<?x?x16x16xi32>) -> tensor<?x?x16x16xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x2xi8>, tensor<?x?x16x2xi8>)
      outs(%arg2 : tensor<?x?x16x16xi32>) -> tensor<?x?x16x16xi32>
  return %0 : tensor<?x?x16x16xi32>
}
// CHECK-LABEL: func @mmt4d_i8i8i32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi8>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x16x16xi32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2_i32:.+]] = arith.constant 2 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C16_i32]], %[[C16_i32]], %[[C2_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

func.func @mmt4d_i8i4i32(%arg0 : tensor<?x?x4x16xi8>, %arg1 : tensor<?x?x8x16xi4>,
    %arg2 : tensor<?x?x4x8xi32>) -> tensor<?x?x4x8xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="aarch64-xyz-xyz", cpu_features="+i8mm"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x4x16xi8>, tensor<?x?x8x16xi4>)
      outs(%arg2 : tensor<?x?x4x8xi32>) -> tensor<?x?x4x8xi32>
  return %0 : tensor<?x?x4x8xi32>
}
// CHECK-LABEL: func @mmt4d_i8i4i32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x4x16xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x8x16xi4>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x4x8xi32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C4_i32:.+]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[C8_i32:.+]] = arith.constant 8 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C4_i32]], %[[C8_i32]], %[[C16_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

func.func @mmt4d_i16i16i32(%arg0 : tensor<?x?x16x2xi16>, %arg1 : tensor<?x?x16x2xi16>,
    %arg2 : tensor<?x?x16x16xi32>) -> tensor<?x?x16x16xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x2xi16>, tensor<?x?x16x2xi16>)
      outs(%arg2 : tensor<?x?x16x16xi32>) -> tensor<?x?x16x16xi32>
  return %0 : tensor<?x?x16x16xi32>
}
// CHECK-LABEL: func @mmt4d_i16i16i32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x16x2xi16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x16x16xi32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2_i32:.+]] = arith.constant 2 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C16_i32]], %[[C16_i32]], %[[C2_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

func.func @mmt4d_f16f16f16(%arg0 : tensor<?x?x16x1xf16>, %arg1 : tensor<?x?x16x1xf16>,
    %arg2 : tensor<?x?x16x16xf16>) -> tensor<?x?x16x16xf16> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x1xf16>, tensor<?x?x16x1xf16>)
      outs(%arg2 : tensor<?x?x16x16xf16>) -> tensor<?x?x16x16xf16>
  return %0 : tensor<?x?x16x16xf16>
}
// CHECK-LABEL: func @mmt4d_f16f16f16(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x16x1xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf16>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 1796 : i32
//  NOSKIPROUND-DAG:   %[[FLAGS:.+]] = arith.constant 769 : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C1_i32:.+]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C16_i32]], %[[C16_i32]], %[[C1_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

func.func @mmt4d_bf16bf16f32(%arg0 : tensor<?x?x16x2xbf16>, %arg1 : tensor<?x?x16x2xbf16>,
    %arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512bf16"}>
} {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<?x?x16x2xbf16>, tensor<?x?x16x2xbf16>)
      outs(%arg2 : tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}
// CHECK-LABEL: func @mmt4d_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x16x2xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x16x2xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x16x16xf32>
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2_i32:.+]] = arith.constant 2 : i32
//  CHECK-DAG:   %[[C16_i32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]], %[[C16_i32]], %[[C16_i32]], %[[C2_i32]], %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

// CHECK-LABEL: func @pack_i8i8_x86(
//       CHECK: ukernel.generic "iree_uk_pack"
func.func @pack_i8i8_x86(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?x7x8xi8>, %arg2 : i8) -> tensor<?x?x7x8xi8> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
} {
  %result = linalg.pack %arg0 padding_value(%arg2 : i8) inner_dims_pos = [0, 1] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xi8> -> tensor<?x?x7x8xi8>
  func.return %result : tensor<?x?x7x8xi8>
}

// -----

// CHECK-LABEL: func @pack_i8i8(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi8>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xi8>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: i8
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 2 : i32
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[ARG2]] : i8 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_i8i8(%arg0 : tensor<?x?xi8>, %arg1 : tensor<?x?x7x8xi8>, %arg2 : i8) -> tensor<?x?x7x8xi8> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.pack %arg0 padding_value(%arg2 : i8) inner_dims_pos = [0, 1] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xi8> -> tensor<?x?x7x8xi8>
  func.return %result : tensor<?x?x7x8xi8>
}

// -----

// CHECK-LABEL: func @pack_f16f16(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: f16
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[BITCAST:.+]] = arith.bitcast %[[ARG2]] : f16 to i16
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[BITCAST]] : i16 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_f16f16(%arg0 : tensor<?x?xf16>, %arg1 : tensor<?x?x7x8xf16>, %arg2 : f16) -> tensor<?x?x7x8xf16> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.pack %arg0 padding_value(%arg2 : f16) inner_dims_pos = [0, 1] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xf16> -> tensor<?x?x7x8xf16>
  func.return %result : tensor<?x?x7x8xf16>
}

// -----

// CHECK-LABEL: func @pack_bf16bf16(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: bf16
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 5 : i32
//  CHECK-DAG:   %[[BITCAST:.+]] = arith.bitcast %[[ARG2]] : bf16 to i16
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[BITCAST]] : i16 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_bf16bf16(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?x7x8xbf16>, %arg2 : bf16) -> tensor<?x?x7x8xbf16> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.pack %arg0 padding_value(%arg2 : bf16) inner_dims_pos = [0, 1] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xbf16> -> tensor<?x?x7x8xbf16>
  func.return %result : tensor<?x?x7x8xbf16>
}

// -----

// CHECK-LABEL: func @pack_i32i32_transpose_inner(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: i32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[ARG2]] : i32 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_i32i32_transpose_inner(%arg0 : tensor<?x?xi32>, %arg1 : tensor<?x?x7x8xi32>, %arg2 : i32) -> tensor<?x?x7x8xi32> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.pack %arg0 padding_value(%arg2 : i32) inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xi32> -> tensor<?x?x7x8xi32>
  func.return %result : tensor<?x?x7x8xi32>
}

// -----

// CHECK-LABEL: func @pack_f32f32_transpose_inner_and_outer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x7x8xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: f32
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[BITCAST:.+]] = arith.bitcast %[[ARG2]] : f32 to i32
//  CHECK-DAG:   %[[PAD:.+]] = arith.extui %[[BITCAST]] : i32 to i64
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[OUT_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.pack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[OUT_SIZE2]], %[[OUT_SIZE3]], %[[PAD]], %[[FLAGS]] :
func.func @pack_f32f32_transpose_inner_and_outer(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?x7x8xf32>, %arg2 : f32) -> tensor<?x?x7x8xf32> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.pack %arg0 padding_value(%arg2 : f32) outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?xf32> -> tensor<?x?x7x8xf32>
  func.return %result : tensor<?x?x7x8xf32>
}

// -----

// Check that linalg.pack is not lowered to a microkernel by default - it should
// only be on VMVX.
// CHECK-LABEL: func @unpack_f16f16_default(
//       CHECK:   linalg.unpack
func.func @unpack_f16f16_default(%arg0 : tensor<?x?x7x8xf16>, %arg1 : tensor<?x?xf16>) -> tensor<?x?xf16> {
  %result = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [7, 8] into %arg1
      : tensor<?x?x7x8xf16> -> tensor<?x?xf16>
  func.return %result : tensor<?x?xf16>
}

// -----

// CHECK-LABEL: func @unpack_f16f16(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x7x8xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf16>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[IN_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[IN_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.unpack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[IN_SIZE2]], %[[IN_SIZE3]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[FLAGS]] :
func.func @unpack_f16f16(%arg0 : tensor<?x?x7x8xf16>, %arg1 : tensor<?x?xf16>) -> tensor<?x?xf16> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [7, 8] into %arg1
      : tensor<?x?x7x8xf16> -> tensor<?x?xf16>
  func.return %result : tensor<?x?xf16>
}

// -----

// CHECK-LABEL: func @unpack_i32i32_transpose_inner(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x7x8xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xi32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[IN_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[IN_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.unpack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[IN_SIZE2]], %[[IN_SIZE3]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[FLAGS]] :
func.func @unpack_i32i32_transpose_inner(%arg0 : tensor<?x?x7x8xi32>, %arg1 : tensor<?x?xi32>) -> tensor<?x?xi32> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.unpack %arg0 inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?x7x8xi32> -> tensor<?x?xi32>
  func.return %result : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @unpack_f32f32_transpose_inner_and_outer(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x7x8xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//  CHECK-DAG:   %[[IN_SIZE0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[IN_SIZE1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[OUT_SIZE0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[OUT_SIZE1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[IN_SIZE2:.+]] = arith.constant 7 : index
//  CHECK-DAG:   %[[IN_SIZE3:.+]] = arith.constant 8 : index
//       CHECK: ukernel.generic "vmvx.unpack"
//  CHECK-SAME:   ins(%[[ARG0]] :
//  CHECK-SAME:   outs(%[[ARG1]] :
//  CHECK-SAME:   (%[[IN_SIZE0]], %[[IN_SIZE1]], %[[IN_SIZE2]], %[[IN_SIZE3]], %[[OUT_SIZE0]], %[[OUT_SIZE1]], %[[FLAGS]] :
func.func @unpack_f32f32_transpose_inner_and_outer(%arg0 : tensor<?x?x7x8xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result = linalg.unpack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [7, 8] into %arg1
      : tensor<?x?x7x8xf32> -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @query_tile_sizes_2d(
//   CHECK-DAG:   %[[DYNAMIC:.+]] = arith.constant -9223372036854775808 : index
//   CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//       CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic "vmvx.query_tile_sizes.2d"
//  CHECK-SAME:     ins(%[[DYNAMIC]], %[[DYNAMIC]], %[[FLAGS]] : index, index, i32)
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1 : index, index
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @query_tile_sizes_2d() -> (index, index)  attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result:2 = iree_codegen.query_tile_sizes tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> index, index
  return %result#0, %result#1 : index, index
}

// -----

// CHECK-LABEL: func @query_tile_sizes_2d_with_layouts(
//   CHECK-DAG:   %[[DYNAMIC:.+]] = arith.constant -9223372036854775808 : index
//   CHECK-DAG:   %[[FLAGS:.+]] = arith.constant {{[0-9]+}} : i32
//       CHECK:   %[[RESULT:.+]]:2 = iree_codegen.ukernel.generic "vmvx.query_tile_sizes.2d"
//  CHECK-SAME:     ins(%[[DYNAMIC]], %[[DYNAMIC]], %[[FLAGS]] : index, index, i32)
//       CHECK:   return %[[RESULT]]#0, %[[RESULT]]#1 : index, index
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_attr = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32]>
#encoding = #iree_encoding.layout<[#iree_cpu.vmvx_encoding_resolver<configuration = {encoding_attr = #encoding_attr, encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [-9223372036854775808, -9223372036854775808], outerDimsPerm = [0, 1]}}>]>
func.func @query_tile_sizes_2d_with_layouts() -> (index, index)  attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %result:2 = iree_codegen.query_tile_sizes tensor<?x?xf32, #encoding> -> index, index
  return %result#0, %result#1 : index, index
}

// -----

func.func @mmt4d_i16u4i32_extend_producers(%arg0: tensor<10x10x1x8xi16>, %arg1: tensor<10x10x32x8xi4>, %arg2: tensor<10x10x1x32xi32>) -> tensor<10x10x1x32xi32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple="x86_64-xyz-xyz", cpu_features="+avx512vnni"}>
} {
  %0 = tensor.empty() : tensor<10x10x1x8xi32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
                        ins(%arg0 : tensor<10x10x1x8xi16>) outs(%0 : tensor<10x10x1x8xi32>) {
  ^bb0(%in: i16, %out: i32):
    %5 = arith.extsi %in : i16 to i32
    linalg.yield %5 : i32
  } -> tensor<10x10x1x8xi32>
  %2 = tensor.empty() : tensor<10x10x32x8xi32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
                        ins(%arg1 : tensor<10x10x32x8xi4>) outs(%2 : tensor<10x10x32x8xi32>) {
  ^bb0(%in: i4, %out: i32):
    %5 = arith.extui %in : i4 to i32
    linalg.yield %5 : i32
  } -> tensor<10x10x32x8xi32>
  %4 = linalg.mmt4d ins(%1, %3 : tensor<10x10x1x8xi32>, tensor<10x10x32x8xi32>) outs(%arg2 : tensor<10x10x1x32xi32>) -> tensor<10x10x1x32xi32>
  return %4 : tensor<10x10x1x32xi32>
}
// CHECK-LABEL: func @mmt4d_i16u4i32_extend_producers(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<10x10x1x8xi16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<10x10x32x8xi4>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<10x10x1x32xi32>
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_mmt4d"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>
func.func @conv_nchwc_f32f32f32(%input: tensor<1x1x16x16x16xf32>, %filter: tensor<1x1x3x3x16x16xf32>, %output: tensor<1x1x14x14x16xf32>) -> tensor<1x1x14x14x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f"}>
} {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} ins(%input, %filter : tensor<1x1x16x16x16xf32>, tensor<1x1x3x3x16x16xf32>) outs(%output : tensor<1x1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x14x14x16xf32>
  return %0 : tensor<1x1x14x14x16xf32>
}
// CHECK-LABEL: func @conv_nchwc_f32f32f32(
// CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x1x16x16x16xf32>
// CHECK-SAME:     %[[FILTER:[a-zA-Z0-9]+]]: tensor<1x1x3x3x16x16xf32>
// CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<1x1x14x14x16xf32>
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//  CHECK-DAG:   %[[C14:.+]] = arith.constant 14 : index
//  CHECK-DAG:   %[[C16_I32:.+]] = arith.constant 16 : i32
//  CHECK-DAG:   %[[C1_I32:.+]] = arith.constant 1 : i32
// Flag constant: 769 = 0x301 = ACCUMULATE (0x100) | ALLOW_GENERIC_FALLBACK_TILE_FUNCTION (0x200) | TYPE_F32F32F32 (0x1)
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 769 : i32
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_conv_nchwc"
// CHECK-SAME:       ins(%[[INPUT]], %[[FILTER]] :
// CHECK-SAME:       outs(%[[OUTPUT]] :
// CHECK-SAME:       (%[[C1]], %[[C1]], %[[C14]], %[[C14]], %[[C1]], %[[C3]], %[[C3]], %[[C16_I32]], %[[C16_I32]], %[[C1_I32]], %[[C1_I32]], %[[FLAGS]] :
// CHECK-SAME:       strided_dims({{\[}}[0, 1, 2], [0, 1, 2, 3], [0, 1, 2]])
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

// Zero-initialized accumulator: the linalg.fill is folded away and the ukernel
// writes a fresh buffer.

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>
func.func @conv_nchwc_zero_fill(%input: tensor<1x1x16x16x16xf32>, %filter: tensor<1x1x3x3x16x16xf32>) -> tensor<1x1x14x14x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f"}>
} {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x1x14x14x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x1x14x14x16xf32>) -> tensor<1x1x14x14x16xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} ins(%input, %filter : tensor<1x1x16x16x16xf32>, tensor<1x1x3x3x16x16xf32>) outs(%fill : tensor<1x1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x14x14x16xf32>
  return %0 : tensor<1x1x14x14x16xf32>
}
// CHECK-LABEL: func @conv_nchwc_zero_fill(
// Flag constant: 513 = 0x201 = ALLOW_GENERIC_FALLBACK_TILE_FUNCTION (0x200) | TYPE_F32F32F32 (0x1); no ACCUMULATE since the fill zeroes the accumulator
//  CHECK-DAG:   %[[FLAGS:.+]] = arith.constant 513 : i32
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x14x14x16xf32>
//      CHECK:   %[[MICRO_KERNEL:.+]]:2 = iree_codegen.ukernel.generic "iree_uk_conv_nchwc"
// CHECK-SAME:       outs(%[[EMPTY]] :
// CHECK-SAME:       %[[FLAGS]] :
//      CHECK:   return %[[MICRO_KERNEL]]#0

// -----

// Dynamic inner tiles (e.g. scalable c0/k0) flow to the ukernel as runtime
// tensor.dim/index_cast k0/c0 operands instead of baked constants.

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>
func.func @conv_nchwc_dynamic_tiles(%input: tensor<1x1x16x16x?xf32>, %filter: tensor<1x1x3x3x?x?xf32>, %output: tensor<1x1x14x14x?xf32>) -> tensor<1x1x14x14x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f"}>
} {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} ins(%input, %filter : tensor<1x1x16x16x?xf32>, tensor<1x1x3x3x?x?xf32>) outs(%output : tensor<1x1x14x14x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x14x14x?xf32>
  return %0 : tensor<1x1x14x14x?xf32>
}
// CHECK-LABEL: func @conv_nchwc_dynamic_tiles(
// CHECK-SAME:     %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x1x16x16x?xf32>
// CHECK-SAME:     %[[FILTER:[a-zA-Z0-9]+]]: tensor<1x1x3x3x?x?xf32>
// CHECK-SAME:     %[[OUTPUT:[a-zA-Z0-9]+]]: tensor<1x1x14x14x?xf32>
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : index
//      CHECK:   %[[K0_DIM:.+]] = tensor.dim %[[FILTER]], %[[C5]]
//      CHECK:   %[[K0:.+]] = arith.index_cast %[[K0_DIM]] : index to i32
//      CHECK:   %[[C0_DIM:.+]] = tensor.dim %[[FILTER]], %[[C4]]
//      CHECK:   %[[C0:.+]] = arith.index_cast %[[C0_DIM]] : index to i32
//      CHECK:   iree_codegen.ukernel.generic "iree_uk_conv_nchwc"
// CHECK-SAME:       ins(%[[INPUT]], %[[FILTER]] :
// CHECK-SAME:       outs(%[[OUTPUT]] :
// CHECK-SAME:       %[[K0]], %[[C0]],

// -----

// Without a ukernel target attribute the conv generic is left untouched.

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>
func.func @negative_conv_ukernel(%input: tensor<1x1x16x16x16xf32>, %filter: tensor<1x1x3x3x16x16xf32>, %output: tensor<1x1x14x14x16xf32>) -> tensor<1x1x14x14x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f"}>
} {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} ins(%input, %filter : tensor<1x1x16x16x16xf32>, tensor<1x1x3x3x16x16xf32>) outs(%output : tensor<1x1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x14x14x16xf32>
  return %0 : tensor<1x1x14x14x16xf32>
}
// CHECK-LABEL: func @negative_conv_ukernel(
//  CHECK-NOT:   iree_uk_conv_nchwc
//      CHECK:   linalg.generic

// -----

// linalg.generic with non-MAC body is not recognized as convolution and left as is.

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d4, d2 + d5, d3 + d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d4, d5, d6, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d7)>
func.func @negative_conv_ukernel_non_conv_body(%input: tensor<1x1x16x16x16xf32>, %filter: tensor<1x1x3x3x16x16xf32>, %output: tensor<1x1x14x14x16xf32>) -> tensor<1x1x14x14x16xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {ukernels = "all", target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f"}>
} {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]} ins(%input, %filter : tensor<1x1x16x16x16xf32>, tensor<1x1x3x3x16x16xf32>) outs(%output : tensor<1x1x14x14x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.subf %in, %in_0 : f32
    %2 = arith.addf %1, %out : f32
    linalg.yield %2 : f32
  } -> tensor<1x1x14x14x16xf32>
  return %0 : tensor<1x1x14x14x16xf32>
}
// CHECK-LABEL: func @negative_conv_ukernel_non_conv_body(
//  CHECK-NOT:   iree_uk_conv_nchwc
//      CHECK:   linalg.generic
