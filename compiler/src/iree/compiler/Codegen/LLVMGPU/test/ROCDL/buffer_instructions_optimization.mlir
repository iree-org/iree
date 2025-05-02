// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 \
// RUN: --pass-pipeline="builtin.module(func.func(iree-rocdl-buffer-instructions-optimization, canonicalize, cse))" %s \
// RUN:  | FileCheck %s

func.func @simplify_mask(%1 : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %index1 : index, %index2 : index) -> vector<1x1x1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c8 : vector<1x1x1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
  return %read : vector<1x1x1x8xbf16>
}

// CHECK-LABEL: @simplify_mask
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
//   CHECK-DAG: %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<1x1x1x8xbf16>
//       CHECK: %[[LHS:.+]] = arith.index_castui %[[ARG1]] : index to i1
//       CHECK: %[[RHS:.+]] = arith.index_castui %[[ARG2]] : index to i1
//       CHECK: %[[AND:.+]] = arith.andi %[[LHS]], %[[RHS]] : i1
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK: %[[SEL:.+]] = arith.select %[[AND]], %[[READ]], %[[CST]] : vector<1x1x1x8xbf16>
//       CHECK: return %[[SEL]] : vector<1x1x1x8xbf16>

// -----

func.func @simplify_mask2(%1 : memref<?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %index1 : index) -> vector<1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %index1, %c8 : vector<1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : memref<?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x8xbf16>
  return %read : vector<1x8xbf16>
}

// CHECK-LABEL: @simplify_mask2
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: index)
//   CHECK-DAG: %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<1x8xbf16>
//       CHECK: %[[COND:.+]] = arith.index_castui %[[ARG1]] : index to i1
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK: %[[SEL:.+]] = arith.select %[[COND]], %[[READ]], %[[CST]] : vector<1x8xbf16>
//       CHECK: return %[[SEL]] : vector<1x8xbf16>

// -----

func.func @simplify_mask3(%1 : memref<?x?x1x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %index1 : index, %index2 : index, %index3 : index) -> vector<1x1x1x1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %index1, %index2, %c1, %index3, %c8 : vector<1x1x1x1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true, true]} : memref<?x?x1x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x1x8xbf16>
  return %read : vector<1x1x1x1x8xbf16>
}

// CHECK-LABEL: @simplify_mask3
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<?x?x1x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
//   CHECK-DAG: %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<1x1x1x1x8xbf16>
//   CHECK-DAG: %[[LHS:.+]] = arith.index_castui %[[ARG1]] : index to i1
//   CHECK-DAG: %[[MID:.+]] = arith.index_castui %[[ARG2]] : index to i1
//       CHECK: %[[AND1:.+]] = arith.andi %[[LHS]], %[[MID]] : i1
//   CHECK-DAG: %[[RHS:.+]] = arith.index_castui %[[ARG3]] : index to i1
//       CHECK: %[[AND2:.+]] = arith.andi %[[AND1]], %[[RHS]] : i1
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK: %[[SEL:.+]] = arith.select %[[AND2]], %[[READ]], %[[CST]] : vector<1x1x1x1x8xbf16>
//       CHECK: return %[[SEL]] : vector<1x1x1x1x8xbf16>

// -----

func.func @simplify_mask4(%1 : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %2 : vector<1x1x1x8xbf16>,
  %3 : vector<1x1x1x8xbf16>, %index1 : index, %index2 : index) -> vector<1x1x1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c8 : vector<1x1x1x8xi1>
  vector.transfer_write %2, %1[%c0, %c0, %c0, %c0], %mask {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
  vector.transfer_write %3, %1[%c0, %c0, %c0, %c0], %mask {in_bounds = [true, true, true, true]} : vector<1x1x1x8xbf16>, memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>
  return %read : vector<1x1x1x8xbf16>
}

// CHECK-LABEL: @simplify_mask4
//  CHECK-SAME:  (%[[ARG0:.+]]: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: vector<1x1x1x8xbf16>, %[[ARG2:.+]]: vector<1x1x1x8xbf16>, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index)
//   CHECK-DAG: %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<1x1x1x8xbf16>
//   CHECK-DAG: %[[CSTBF16:.+]] = arith.constant 1.000000e+00 : bf16
//   CHECK-DAG: %[[MASK:.+]] = vector.create_mask %{{.+}}, %[[ARG3]], %[[ARG4]], %{{.+}} : vector<1x1x1x8xi1>
//       CHECK: vector.transfer_write %[[ARG1]], %[[ARG0]]{{.+}}, %[[MASK]]
//       CHECK: %[[IDX1:.+]] = arith.index_castui %[[ARG3]] : index to i1
//       CHECK: %[[IDX2:.+]] = arith.index_castui %[[ARG4]] : index to i1
//       CHECK: %[[AND:.+]] = arith.andi %[[IDX1]], %[[IDX2]] : i1
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]{{.+}}, %[[CSTBF16]]
//       CHECK: %[[SEL:.+]] = arith.select %[[AND]], %[[READ]], %[[CST]]
//       CHECK: vector.transfer_write %[[ARG2]], %[[ARG0]]{{.+}}, %[[MASK]]
//       CHECK: return %[[SEL]]

// -----

func.func @simplify_mask5(%1 : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %2 : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>,
    %index1 : index, %index2 : index) -> (vector<1x1x1x8xbf16>, vector<1x1x1x8xbf16>)  {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c8 : vector<1x1x1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
  %read2 = vector.transfer_read %2[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
  return %read, %read2 : vector<1x1x1x8xbf16>,  vector<1x1x1x8xbf16>
}

// CHECK-LABEL: @simplify_mask5
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG2:.+]]: index, %[[ARG3:.+]]: index)
//   CHECK-DAG: %[[CST:.+]] = arith.constant dense<1.000000e+00> : vector<1x1x1x8xbf16>
//   CHECK-DAG: %[[CSTBF16:.+]] = arith.constant 1.000000e+00 : bf16
//       CHECK: %[[IDX1:.+]] = arith.index_castui %[[ARG2]] : index to i1
//       CHECK: %[[IDX2:.+]] = arith.index_castui %[[ARG3]] : index to i1
//       CHECK: %[[AND:.+]] = arith.andi %[[IDX1]], %[[IDX2]] : i1
//       CHECK: %[[READ0:.+]] = vector.transfer_read %[[ARG0]]{{.+}}, %[[CSTBF16]]
//       CHECK: %[[SEL0:.+]] = arith.select %[[AND]], %[[READ0]], %[[CST]]
//       CHECK: %[[READ1:.+]] = vector.transfer_read %[[ARG1]]{{.+}}, %[[CSTBF16]]
//       CHECK: %[[SEL1:.+]] = arith.select %[[AND]], %[[READ1]], %[[CST]]
//       CHECK: return %[[SEL0]], %[[SEL1]]

// -----

func.func @no_simplify_mask_no_fat_raw_buffer(%1 : memref<1x?x?x8xbf16>, %index1 : index, %index2 : index) -> vector<1x1x1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c8 : vector<1x1x1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16>, vector<1x1x1x8xbf16>
  return %read : vector<1x1x1x8xbf16>
}

// CHECK-LABEL: @no_simplify_mask_no_fat_raw_buffer
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<1x?x?x8xbf16>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
//   CHECK-DAG: %[[MASK:.+]] = vector.create_mask
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//  CHECK-SAME: %[[MASK]]
//       CHECK: return %[[READ]] : vector<1x1x1x8xbf16>

// -----

func.func @no_simplify_mask_tensor(%1 : tensor<1x?x?x8xbf16>, %index1 : index, %index2 : index) -> vector<1x1x1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c8 : vector<1x1x1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : tensor<1x?x?x8xbf16>, vector<1x1x1x8xbf16>
  return %read : vector<1x1x1x8xbf16>
}

// CHECK-LABEL: @no_simplify_mask_tensor
//  CHECK-SAME:   (%[[ARG0:.+]]: tensor<1x?x?x8xbf16>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
//   CHECK-DAG: %[[MASK:.+]] = vector.create_mask
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//  CHECK-SAME: %[[MASK]]
//       CHECK: return %[[READ]] : vector<1x1x1x8xbf16>

// -----

func.func @no_simplify_mask_outofbounds(%1 : memref<1x?x?x6xbf16, #amdgpu.address_space<fat_raw_buffer>>, %index1 : index, %index2 : index) -> vector<1x1x1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c8 : vector<1x1x1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, false]} : memref<1x?x?x6xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
  return %read : vector<1x1x1x8xbf16>
}

// CHECK-LABEL: @no_simplify_mask_outofbounds
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<1x?x?x6xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
//   CHECK-DAG: %[[MASK:.+]] = vector.create_mask
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//  CHECK-SAME: %[[MASK]]
//       CHECK: return %[[READ]] : vector<1x1x1x8xbf16>

// -----

func.func @no_simplify_partial_mask(%1 : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %index1 : index, %index2 : index) -> vector<1x1x1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c6 : vector<1x1x1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x1x8xbf16>
  return %read : vector<1x1x1x8xbf16>
}

// CHECK-LABEL: @no_simplify_partial_mask
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
//   CHECK-DAG: %[[MASK:.+]] = vector.create_mask
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//  CHECK-SAME: %[[MASK]]
//       CHECK: return %[[READ]] : vector<1x1x1x8xbf16>

// -----

func.func @no_simplify_mask_nonunit(%1 : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %index1 : index, %index2 : index) -> vector<1x1x2x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %index1, %index2, %c8 : vector<1x1x2x8xi1>
  %read = vector.transfer_read %1[%c0, %c0, %c0, %c0], %cst, %mask {in_bounds = [true, true, true, true]} : memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x1x2x8xbf16>
  return %read : vector<1x1x2x8xbf16>
}

// CHECK-LABEL: @no_simplify_mask_nonunit
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<1x?x?x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, %[[ARG1:.+]]: index, %[[ARG2:.+]]: index)
//   CHECK-DAG: %[[MASK:.+]] = vector.create_mask
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//  CHECK-SAME: %[[MASK]]
//       CHECK: return %[[READ]] : vector<1x1x2x8xbf16>

// -----

// This type of simplification is taken care of by canonicalization directly, the test just shows
// that we didnt break that behavior.
func.func @simplify_trivial(%1 : memref<1x8xbf16, #amdgpu.address_space<fat_raw_buffer>>) -> vector<1x8xbf16> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 1.000000e+00 : bf16
  %mask = vector.create_mask %c1, %c8 : vector<1x8xi1>
  %read = vector.transfer_read %1[%c0, %c0], %cst, %mask {in_bounds = [true, true]} : memref<1x8xbf16, #amdgpu.address_space<fat_raw_buffer>>, vector<1x8xbf16>
  return %read : vector<1x8xbf16>
}

// CHECK-LABEL: @simplify_trivial
//  CHECK-SAME:   (%[[ARG0:.+]]: memref<1x8xbf16, #amdgpu.address_space<fat_raw_buffer>>)
//   CHECK-NOT: vector.create_mask
//       CHECK: %[[READ:.+]] = vector.transfer_read %[[ARG0]]
//       CHECK: return %[[READ]] : vector<1x8xbf16>
