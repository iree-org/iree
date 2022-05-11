// RUN: iree-opt --iree-llvmcpu-vector-contract-custom-kernels='arch=aarch64' %s | FileCheck %s --check-prefix=AARCH64-BASELINE
// RUN: iree-opt --iree-llvmcpu-vector-contract-custom-kernels='arch=aarch64 features=+dotprod' %s | FileCheck %s --check-prefix=AARCH64-DOTPROD
// RUN: iree-opt --iree-llvmcpu-vector-contract-custom-kernels='arch=aarch64 features=+i8mm' %s | FileCheck %s --check-prefix=AARCH64-I8MM

// There are two parts to this test: the "deep" part and the "wide part".

//////////////////////////////////////////////////////////////////////////////
// The "deep part": test one case in depth.
//////////////////////////////////////////////////////////////////////////////

// -----
func.func @check_in_depth_mmt_8x4x8_i8i8i32(
    %lhs: vector<8x4xi8>,
    %rhs: vector<8x4xi8>,
    %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x4xi8> to vector<8x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x4xi8> to vector<8x4xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}
// AARCH64-DOTPROD-LABEL: func @check_in_depth_mmt_8x4x8_i8i8i32(
// AARCH64-DOTPROD-SAME:      %[[LHS:[^:[:space:]]+]]
// AARCH64-DOTPROD-SAME:      %[[RHS:[^:[:space:]]+]]
// AARCH64-DOTPROD-SAME:      %[[ACC:[^:[:space:]]+]]
// AARCH64-DOTPROD-SAME:        -> vector<8x8xi32> {
// AARCH64-DOTPROD-DAG:     %[[INITRES:.+]] = arith.constant dense<0> : vector<64xi32>
// AARCH64-DOTPROD-DAG:     %[[LHS1D:.+]] = vector.shape_cast %[[LHS]] : vector<8x4xi8> to vector<32xi8>
// AARCH64-DOTPROD-DAG:     %[[RHS1D:.+]] = vector.shape_cast %[[RHS]] : vector<8x4xi8> to vector<32xi8>
// AARCH64-DOTPROD-DAG:     %[[ACC1D:.+]] = vector.shape_cast %[[ACC]] : vector<8x8xi32> to vector<64xi32>
// AARCH64-DOTPROD-DAG:     %[[LHS1D_0:.+]] = vector.extract_strided_slice %[[LHS1D]] {offsets = [0], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// AARCH64-DOTPROD-DAG:     %[[RHS1D_0:.+]] = vector.extract_strided_slice %[[RHS1D]] {offsets = [0], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// AARCH64-DOTPROD-DAG:     %[[LHS1D_1:.+]] = vector.extract_strided_slice %[[LHS1D]] {offsets = [16], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// AARCH64-DOTPROD-DAG:     %[[RHS1D_1:.+]] = vector.extract_strided_slice %[[RHS1D]] {offsets = [16], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_0:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [0], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_1:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [4], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_2:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [8], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_3:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [12], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_4:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [16], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_5:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [20], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_6:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [24], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_7:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [28], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_8:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [32], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_9:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [36], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_10:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [40], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_11:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [44], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_12:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [48], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_13:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [52], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_14:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [56], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ACC1D_15:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [60], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// AARCH64-DOTPROD-DAG:     %[[ASM:.+]] = llvm.inline_asm asm_dialect = att
// AARCH64-DOTPROD-SAME:      {{((.*sdot){16})}}
// AARCH64-DOTPROD-SAME:      "{{(\=w,){16}(w,){4}0,1,.*,15}}"
// AARCH64-DOTPROD-SAME:      {{\((vector<16xi8>, ){4}(vector<4xi32>(, )?){16}\)}}
// AARCH64-DOTPROD-SAME:      ->  !llvm.struct<({{((vector<4xi32>(, )?){16})}})>
// AARCH64-DOTPROD-DAG:         %[[RES0:.+]] = llvm.extractvalue %[[ASM]][0]
// AARCH64-DOTPROD-DAG:         %[[RES1:.+]] = llvm.extractvalue %[[ASM]][1]
// AARCH64-DOTPROD-DAG:         %[[RES2:.+]] = llvm.extractvalue %[[ASM]][2]
// AARCH64-DOTPROD-DAG:         %[[RES3:.+]] = llvm.extractvalue %[[ASM]][3]
// AARCH64-DOTPROD-DAG:         %[[RES4:.+]] = llvm.extractvalue %[[ASM]][4]
// AARCH64-DOTPROD-DAG:         %[[RES5:.+]] = llvm.extractvalue %[[ASM]][5]
// AARCH64-DOTPROD-DAG:         %[[RES6:.+]] = llvm.extractvalue %[[ASM]][6]
// AARCH64-DOTPROD-DAG:         %[[RES7:.+]] = llvm.extractvalue %[[ASM]][7]
// AARCH64-DOTPROD-DAG:         %[[RES8:.+]] = llvm.extractvalue %[[ASM]][8]
// AARCH64-DOTPROD-DAG:         %[[RES9:.+]] = llvm.extractvalue %[[ASM]][9]
// AARCH64-DOTPROD-DAG:         %[[RES10:.+]] = llvm.extractvalue %[[ASM]][10]
// AARCH64-DOTPROD-DAG:         %[[RES11:.+]] = llvm.extractvalue %[[ASM]][11]
// AARCH64-DOTPROD-DAG:         %[[RES12:.+]] = llvm.extractvalue %[[ASM]][12]
// AARCH64-DOTPROD-DAG:         %[[RES13:.+]] = llvm.extractvalue %[[ASM]][13]
// AARCH64-DOTPROD-DAG:         %[[RES14:.+]] = llvm.extractvalue %[[ASM]][14]
// AARCH64-DOTPROD-DAG:         %[[RES15:.+]] = llvm.extractvalue %[[ASM]][15]
// AARCH64-DOTPROD:         %[[INS0:.+]] = vector.insert_strided_slice %[[RES0]], %[[INITRES]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS1:.+]] = vector.insert_strided_slice %[[RES1]], %[[INS0]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS2:.+]] = vector.insert_strided_slice %[[RES2]], %[[INS1]] {offsets = [8], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS3:.+]] = vector.insert_strided_slice %[[RES3]], %[[INS2]] {offsets = [12], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS4:.+]] = vector.insert_strided_slice %[[RES4]], %[[INS3]] {offsets = [16], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS5:.+]] = vector.insert_strided_slice %[[RES5]], %[[INS4]] {offsets = [20], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS6:.+]] = vector.insert_strided_slice %[[RES6]], %[[INS5]] {offsets = [24], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS7:.+]] = vector.insert_strided_slice %[[RES7]], %[[INS6]] {offsets = [28], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS8:.+]] = vector.insert_strided_slice %[[RES8]], %[[INS7]] {offsets = [32], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS9:.+]] = vector.insert_strided_slice %[[RES9]], %[[INS8]] {offsets = [36], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS10:.+]] = vector.insert_strided_slice %[[RES10]], %[[INS9]] {offsets = [40], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS11:.+]] = vector.insert_strided_slice %[[RES11]], %[[INS10]] {offsets = [44], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS12:.+]] = vector.insert_strided_slice %[[RES12]], %[[INS11]] {offsets = [48], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS13:.+]] = vector.insert_strided_slice %[[RES13]], %[[INS12]] {offsets = [52], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS14:.+]] = vector.insert_strided_slice %[[RES14]], %[[INS13]] {offsets = [56], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[INS15:.+]] = vector.insert_strided_slice %[[RES15]], %{{.+}} {offsets = [60], strides = [1]} : vector<4xi32> into vector<64xi32>
// AARCH64-DOTPROD:         %[[RESULT1D:.+]] = vector.shape_cast %[[INS15]] : vector<64xi32> to vector<8x8xi32>
// AARCH64-DOTPROD:         return %[[RESULT1D]] : vector<8x8xi32>

//////////////////////////////////////////////////////////////////////////////
// The "wide part": test that every case picks up the intended asm kernel,
// and some basic checks on the inline_asm, in particular checking the
// generated constraints/clobbers string.
//
// For vector*matrix cases, we simply check that they take the same code path
// as matrix*vector here. The distinction between the two is only in the order
// in which args are passed to the inline_asm, which could only be checked by
// a very in-depth lit test. If a bug was introduced there, that would result
// in e2e test failures. On the other hand, if the vector*matrix -> matrix*vector
// reduction failed to kick in, that would simply be a missed optimization, so
// e2e test would not catch it.
//////////////////////////////////////////////////////////////////////////////

// -----
func.func @mmt_8x1x8_f32f32f32(
    %lhs: vector<8x1xf32>,
    %rhs: vector<8x1xf32>,
    %acc: vector<8x8xf32>) -> vector<8x8xf32> {
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<8x1xf32>, vector<8x1xf32> into vector<8x8xf32>
  return %res : vector<8x8xf32>
}
// AARCH64-BASELINE-LABEL:  @mmt_8x1x8_f32f32f32(
// AARCH64-BASELINE:     llvm.inline_asm
// AARCH64-BASELINE-SAME:      {{((.*fmla){16})}}
// AARCH64-BASELINE-SAME:      "{{(\=w,){16}(w,){4}0,1,.*,15}}"
// AARCH64-BASELINE-SAME:      {{\((vector<4xf32>(, )?){20}\)}}

// -----
func.func @mmt_8x1x1_f32f32f32_matvec(
    %lhs: vector<8x1xf32>,
    %rhs: vector<1x1xf32>,
    %acc: vector<8x1xf32>) -> vector<8x1xf32> {
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<8x1xf32>, vector<1x1xf32> into vector<8x1xf32>
  return %res : vector<8x1xf32>
}
// AARCH64-BASELINE-LABEL:  @mmt_8x1x1_f32f32f32_matvec(
// AARCH64-BASELINE:     llvm.inline_asm
// AARCH64-BASELINE-SAME:      {{((.*fmla){2})}}
// AARCH64-BASELINE-SAME:      "{{(\=w,){2}(w,){3}0,1}}"
// AARCH64-BASELINE-SAME:      {{\((vector<4xf32>, ){2}(f32, ){1}(vector<4xf32>(, )?){2}\)}}

// -----
func.func @mmt_1x1x8_f32f32f32_vecmat(
    %lhs: vector<1x1xf32>,
    %rhs: vector<8x1xf32>,
    %acc: vector<1x8xf32>) -> vector<1x8xf32> {
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<1x1xf32>, vector<8x1xf32> into vector<1x8xf32>
  return %res : vector<1x8xf32>
}
// AARCH64-BASELINE-LABEL:  @mmt_1x1x8_f32f32f32_vecmat(
// AARCH64-BASELINE:     llvm.inline_asm
// AARCH64-BASELINE-SAME:      {{((.*fmla){2})}}
// AARCH64-BASELINE-SAME:      "{{(\=w,){2}(w,){3}0,1}}"
// AARCH64-BASELINE-SAME:      {{\((vector<4xf32>, ){2}(f32, ){1}(vector<4xf32>(, )?){2}\)}}

// -----
func.func @mmt_8x1x8_i8i8i32(
    %lhs: vector<8x1xi8>,
    %rhs: vector<8x1xi8>,
    %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x1xi8> to vector<8x1xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x1xi8> to vector<8x1xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x1xi32>, vector<8x1xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}
// AARCH64-BASELINE-LABEL:  @mmt_8x1x8_i8i8i32(
// AARCH64-BASELINE:     llvm.inline_asm
// AARCH64-BASELINE-SAME:      {{((.*sxtl){2}(.*smlal[ ].*smlal2){8})}}
// AARCH64-BASELINE-SAME:      "{{(\=w,){16}(w,){2}0,1,.*,15}},~{v14},~{v15}"
// AARCH64-BASELINE-SAME:      {{\((vector<8xi8>, ){2}(vector<4xi32>(, )?){16}\)}}

// -----
func.func @mmt_8x8x1_i8i8i32_matvec(
    %lhs: vector<8x8xi8>,
    %rhs: vector<1x8xi8>,
    %acc: vector<8x1xi32>) -> vector<8x1xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x8xi8> to vector<8x8xi32>
  %rhs_wide = arith.extsi %rhs : vector<1x8xi8> to vector<1x8xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x8xi32>, vector<1x8xi32> into vector<8x1xi32>
  return %res : vector<8x1xi32>
}
// AARCH64-BASELINE-LABEL:  @mmt_8x8x1_i8i8i32_matvec(
// AARCH64-BASELINE:     llvm.inline_asm
// AARCH64-BASELINE-SAME:      {{((.*smull[ ].*smull2){4})}}
// AARCH64-BASELINE-SAME:      "{{(\=w,){2}(w,){5}0,1}},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"
// AARCH64-BASELINE-SAME:      {{\((vector<16xi8>, ){4}(vector<8xi8>, ){1}(vector<4xi32>(, )?){2}\)}}

// -----
func.func @mmt_1x8x8_i8i8i32_matvec(
    %lhs: vector<1x8xi8>,
    %rhs: vector<8x8xi8>,
    %acc: vector<1x8xi32>) -> vector<1x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<1x8xi8> to vector<1x8xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x8xi8> to vector<8x8xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<1x8xi32>, vector<8x8xi32> into vector<1x8xi32>
  return %res : vector<1x8xi32>
}
// AARCH64-BASELINE-LABEL:  @mmt_1x8x8_i8i8i32_matvec(
// AARCH64-BASELINE:     llvm.inline_asm
// AARCH64-BASELINE-SAME:      {{((.*smull[ ].*smull2){4})}}
// AARCH64-BASELINE-SAME:      "{{(\=w,){2}(w,){5}0,1}},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15}"
// AARCH64-BASELINE-SAME:      {{\((vector<16xi8>, ){4}(vector<8xi8>, ){1}(vector<4xi32>(, )?){2}\)}}

// -----
func.func @mmt_8x4x8_i8i8i32(
    %lhs: vector<8x4xi8>,
    %rhs: vector<8x4xi8>,
    %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x4xi8> to vector<8x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x4xi8> to vector<8x4xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x4xi32>, vector<8x4xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}
// AARCH64-DOTPROD-LABEL:  @mmt_8x4x8_i8i8i32(
// AARCH64-DOTPROD:     llvm.inline_asm
// AARCH64-DOTPROD-SAME:      {{((.*sdot){16})}}
// AARCH64-DOTPROD-SAME:      "{{(\=w,){16}(w,){4}0,1,.*,15}}"
// AARCH64-DOTPROD-SAME:      {{\((vector<16xi8>, ){4}(vector<4xi32>(, )?){16}\)}}

// -----
func.func @mmt_8x4x1_i8i8i32_matvec(
    %lhs: vector<8x4xi8>,
    %rhs: vector<1x4xi8>,
    %acc: vector<8x1xi32>) -> vector<8x1xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x4xi8> to vector<8x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<1x4xi8> to vector<1x4xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x4xi32>, vector<1x4xi32> into vector<8x1xi32>
  return %res : vector<8x1xi32>
}
// AARCH64-DOTPROD-LABEL:  @mmt_8x4x1_i8i8i32_matvec(
// AARCH64-DOTPROD:     llvm.inline_asm
// AARCH64-DOTPROD-SAME:      {{((.*sdot){2})}}
// AARCH64-DOTPROD-SAME:      "{{(\=w,){2}(w,){3}0,1}}"
// AARCH64-DOTPROD-SAME:      {{\((vector<16xi8>, ){2}(vector<4xi8>, ){1}(vector<4xi32>(, )?){2}\)}}

// -----
func.func @mmt_1x4x8_i8i8i32_vecmat(
    %lhs: vector<1x4xi8>,
    %rhs: vector<8x4xi8>,
    %acc: vector<1x8xi32>) -> vector<1x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<1x4xi8> to vector<1x4xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x4xi8> to vector<8x4xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<1x4xi32>, vector<8x4xi32> into vector<1x8xi32>
  return %res : vector<1x8xi32>
}
// AARCH64-DOTPROD-LABEL:  @mmt_1x4x8_i8i8i32_vecmat(
// AARCH64-DOTPROD:     llvm.inline_asm
// AARCH64-DOTPROD-SAME:      {{((.*sdot){2})}}
// AARCH64-DOTPROD-SAME:      "{{(\=w,){2}(w,){3}0,1}}"
// AARCH64-DOTPROD-SAME:      {{\((vector<16xi8>, ){2}(vector<4xi8>, ){1}(vector<4xi32>(, )?){2}\)}}

// -----
func.func @mmt_8x8x8_i8i8i32(
    %lhs: vector<8x8xi8>,
    %rhs: vector<8x8xi8>,
    %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  %lhs_wide = arith.extsi %lhs : vector<8x8xi8> to vector<8x8xi32>
  %rhs_wide = arith.extsi %rhs : vector<8x8xi8> to vector<8x8xi32>
  %res = vector.contract {
      indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>
      ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
  } %lhs_wide, %rhs_wide, %acc : vector<8x8xi32>, vector<8x8xi32> into vector<8x8xi32>
  return %res : vector<8x8xi32>
}
// AARCH64-I8MM-LABEL:  @mmt_8x8x8_i8i8i32(
// AARCH64-I8MM:     llvm.inline_asm
// AARCH64-I8MM-SAME:      {{((.*smmla){16})}}
// AARCH64-I8MM-SAME:      "{{(\=w,){16}(w,){8}0,1,.*,15}},~{v28},~{v29},~{v30},~{v31}"
// AARCH64-I8MM-SAME:      {{\((vector<16xi8>, ){8}(vector<4xi32>(, )?){16}\)}}
