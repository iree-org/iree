// RUN: iree-opt -iree-llvmcpu-vector-contract-custom-kernels='arch=aarch64 features=+dotprod' %s | FileCheck %s

func @mmt_8x4x8_i8i8i32_aarch64_dotprod_inline_asm(
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
// CHECK-LABEL: func @mmt_8x4x8_i8i8i32_aarch64_dotprod_inline_asm(
// CHECK-SAME:      %[[LHS:[^:[:space:]]+]]
// CHECK-SAME:      %[[RHS:[^:[:space:]]+]]
// CHECK-SAME:      %[[ACC:[^:[:space:]]+]]
// CHECK-SAME:        -> vector<8x8xi32> {
// CHECK-DAG:     %[[INITRES:.+]] = arith.constant dense<0> : vector<64xi32>
// CHECK-DAG:     %[[LHS1D:.+]] = vector.shape_cast %[[LHS]] : vector<8x4xi8> to vector<32xi8>
// CHECK-DAG:     %[[RHS1D:.+]] = vector.shape_cast %[[RHS]] : vector<8x4xi8> to vector<32xi8>
// CHECK-DAG:     %[[ACC1D:.+]] = vector.shape_cast %[[ACC]] : vector<8x8xi32> to vector<64xi32>
// CHECK-DAG:     %[[LHS1D_0:.+]] = vector.extract_strided_slice %[[LHS1D]] {offsets = [0], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK-DAG:     %[[RHS1D_0:.+]] = vector.extract_strided_slice %[[RHS1D]] {offsets = [0], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK-DAG:     %[[LHS1D_1:.+]] = vector.extract_strided_slice %[[LHS1D]] {offsets = [16], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK-DAG:     %[[RHS1D_1:.+]] = vector.extract_strided_slice %[[RHS1D]] {offsets = [16], sizes = [16], strides = [1]} : vector<32xi8> to vector<16xi8>
// CHECK-DAG:     %[[ACC1D_0:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [0], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_1:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [4], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_2:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [8], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_3:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [12], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_4:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [16], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_5:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [20], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_6:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [24], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_7:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [28], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_8:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [32], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_9:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [36], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_10:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [40], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_11:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [44], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_12:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [48], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_13:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [52], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_14:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [56], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ACC1D_15:.+]] = vector.extract_strided_slice %[[ACC1D]] {offsets = [60], sizes = [4], strides = [1]} : vector<64xi32> to vector<4xi32>
// CHECK-DAG:     %[[ASM:.+]] = llvm.inline_asm asm_dialect = att
// CHECK-SAME:      "{{([\0A ]+sdot \$[0-9]+\.4s, \$[0-9]+\.16b, \$[0-9]+\.4b[[0-9]+]){16}[\0A ]+}}"
// CHECK-SAME:      "{{(\=w,){16}(w,){4}0,1,.*,15}}"
// CHECK-SAME:      {{\((vector<16xi8>, ){4}(vector<4xi32>(, )?){16}\)}}
// CHECK-SAME:      ->  !llvm.struct<({{((vector<4xi32>(, )?){16})}})>
// CHECK-DAG:         %[[RES0:.+]] = llvm.extractvalue %[[ASM]][0]
// CHECK-DAG:         %[[RES1:.+]] = llvm.extractvalue %[[ASM]][1]
// CHECK-DAG:         %[[RES2:.+]] = llvm.extractvalue %[[ASM]][2]
// CHECK-DAG:         %[[RES3:.+]] = llvm.extractvalue %[[ASM]][3]
// CHECK-DAG:         %[[RES4:.+]] = llvm.extractvalue %[[ASM]][4]
// CHECK-DAG:         %[[RES5:.+]] = llvm.extractvalue %[[ASM]][5]
// CHECK-DAG:         %[[RES6:.+]] = llvm.extractvalue %[[ASM]][6]
// CHECK-DAG:         %[[RES7:.+]] = llvm.extractvalue %[[ASM]][7]
// CHECK-DAG:         %[[RES8:.+]] = llvm.extractvalue %[[ASM]][8]
// CHECK-DAG:         %[[RES9:.+]] = llvm.extractvalue %[[ASM]][9]
// CHECK-DAG:         %[[RES10:.+]] = llvm.extractvalue %[[ASM]][10]
// CHECK-DAG:         %[[RES11:.+]] = llvm.extractvalue %[[ASM]][11]
// CHECK-DAG:         %[[RES12:.+]] = llvm.extractvalue %[[ASM]][12]
// CHECK-DAG:         %[[RES13:.+]] = llvm.extractvalue %[[ASM]][13]
// CHECK-DAG:         %[[RES14:.+]] = llvm.extractvalue %[[ASM]][14]
// CHECK-DAG:         %[[RES15:.+]] = llvm.extractvalue %[[ASM]][15]
// CHECK:         %[[INS0:.+]] = vector.insert_strided_slice %[[RES0]], %[[INITRES]] {offsets = [0], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS1:.+]] = vector.insert_strided_slice %[[RES1]], %[[INS0]] {offsets = [4], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS2:.+]] = vector.insert_strided_slice %[[RES2]], %[[INS1]] {offsets = [8], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS3:.+]] = vector.insert_strided_slice %[[RES3]], %[[INS2]] {offsets = [12], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS4:.+]] = vector.insert_strided_slice %[[RES4]], %[[INS3]] {offsets = [16], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS5:.+]] = vector.insert_strided_slice %[[RES5]], %[[INS4]] {offsets = [20], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS6:.+]] = vector.insert_strided_slice %[[RES6]], %[[INS5]] {offsets = [24], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS7:.+]] = vector.insert_strided_slice %[[RES7]], %[[INS6]] {offsets = [28], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS8:.+]] = vector.insert_strided_slice %[[RES8]], %[[INS7]] {offsets = [32], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS9:.+]] = vector.insert_strided_slice %[[RES9]], %[[INS8]] {offsets = [36], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS10:.+]] = vector.insert_strided_slice %[[RES10]], %[[INS9]] {offsets = [40], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS11:.+]] = vector.insert_strided_slice %[[RES11]], %[[INS10]] {offsets = [44], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS12:.+]] = vector.insert_strided_slice %[[RES12]], %[[INS11]] {offsets = [48], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS13:.+]] = vector.insert_strided_slice %[[RES13]], %[[INS12]] {offsets = [52], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS14:.+]] = vector.insert_strided_slice %[[RES14]], %[[INS13]] {offsets = [56], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[INS15:.+]] = vector.insert_strided_slice %[[RES15]], %{{.+}} {offsets = [60], strides = [1]} : vector<4xi32> into vector<64xi32>
// CHECK:         %[[RESULT1D:.+]] = vector.shape_cast %[[INS15]] : vector<64xi32> to vector<8x8xi32>
// CHECK:         return %[[RESULT1D]] : vector<8x8xi32>
