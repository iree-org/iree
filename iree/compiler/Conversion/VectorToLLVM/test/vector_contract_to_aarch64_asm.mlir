// RUN: iree-opt -iree-codegen-vector-to-aarch64-inline-asm %s | IreeFileCheck %s

func @vector_matmul_to_aarch64_asm_vec_dot(%lhs: memref<4x4xi8>, %rhs: memref<4x4xi8>, %dst: memref<4x4xi32>) {
    %c0 = constant 0 : index
    %cst_i8_0 = constant 0 : i8
    %cst_i32_0 = constant 0 : i32
    %0 = vector.transfer_read %lhs[%c0, %c0] , %cst_i8_0 {masked = [false, false]} : memref<4x4xi8>, vector<4x4xi8>
    %1 = vector.transfer_read %rhs[%c0, %c0] , %cst_i8_0 {masked = [false, false]} : memref<4x4xi8>, vector<4x4xi8>
    %2 = vector.transfer_read %dst[%c0, %c0], %cst_i32_0 {masked = [false, false]} : memref<4x4xi32>, vector<4x4xi32>
    %3 = sexti %0 : vector<4x4xi8> to vector<4x4xi32>
    %4 = sexti %1 : vector<4x4xi8> to vector<4x4xi32>
    %5 = vector.contract {
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d2)>,
            affine_map<(d0, d1, d2) -> (d2, d1)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>
    } %3, %4, %2 : vector<4x4xi32>, vector<4x4xi32> into vector<4x4xi32>
    vector.transfer_write %5, %dst[%c0, %c0] {masked = [false, false]}: vector<4x4xi32>, memref<4x4xi32>
    return
}
// CHECK-LABEL: vector_matmul_to_aarch64_asm_vec_dot
// CHECK: %[[RES_2D:.+]] = constant dense<0> : vector<4x4xi32>
// CHECK: %[[LHS_2D:.+]] = vector.transfer_read
// CHECK: %[[RHS_2d:.+]] = vector.transfer_read
// CHECK: %[[DST:.+]] = vector.transfer_read
// CHECK: %[[DST0:.+]] = vector.extract %[[DST]][0] : vector<4x4xi32>
// CHECK: %[[DST1:.+]] = vector.extract %[[DST]][1] : vector<4x4xi32>
// CHECK: %[[DST2:.+]] = vector.extract %[[DST]][2] : vector<4x4xi32>
// CHECK: %[[DST3:.+]] = vector.extract %[[DST]][3] : vector<4x4xi32>
// CHECK: %[[LHS:.+]] = vector.shape_cast %[[LHS_2D]] : vector<4x4xi8> to vector<16xi8>
// CHECK: %[[RHS_T_2d:.+]] = vector.transpose %[[RHS_2d]], [1, 0]
// CHECK: %[[RHS_T:.+]] = vector.shape_cast %[[RHS_T_2d]] : vector<4x4xi8> to vector<16xi8>
// CHECK: %[[ASM_RESULT:.+]] = llvm.inline_asm {{.*}} "=w,=w,=w,=w,w,w,0,1,2,3" %[[LHS]], %[[RHS_T]], %[[DST0]], %[[DST1]], %[[DST2]], %[[DST3]] 
// CHECK: %[[RES_0:.+]] = llvm.extractvalue %[[ASM_RESULT]][0] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
// CHECK: %[[RES_1:.+]] = llvm.extractvalue %[[ASM_RESULT]][1] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
// CHECK: %[[RES_2:.+]] = llvm.extractvalue %[[ASM_RESULT]][2] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
// CHECK: %[[RES_3:.+]] = llvm.extractvalue %[[ASM_RESULT]][3] : !llvm.struct<(vector<4xi32>, vector<4xi32>, vector<4xi32>, vector<4xi32>)>
// CHECK: %[[RES_2D_0:.+]] = vector.insert %[[RES_0]], %[[RES_2D]] [0] : vector<4xi32> into vector<4x4xi32>
// CHECK: %[[RES_2D_1:.+]] = vector.insert %[[RES_1]], %[[RES_2D_0]] [1] : vector<4xi32> into vector<4x4xi32>
// CHECK: %[[RES_2D_2:.+]] = vector.insert %[[RES_2]], %[[RES_2D_1]] [2] : vector<4xi32> into vector<4x4xi32>
// CHECK: %[[RES_2D_3:.+]] = vector.insert %[[RES_3]], %[[RES_2D_2]] [3] : vector<4xi32> into vector<4x4xi32>