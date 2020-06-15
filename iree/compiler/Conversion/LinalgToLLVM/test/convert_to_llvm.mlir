// RUN: iree-opt -iree-codegen-convert-to-llvm -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: func @convert_dynamic_shape
func @convert_dynamic_shape(%arg0: memref<?x?xf32>, %arg1: memref<2xi32>){
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = load %arg1[%c0] : memref<2xi32>
    %1 = index_cast %0 : i32 to index
    %3 = load %arg1[%c1] : memref<2xi32>
    %4 = index_cast %3 : i32 to index
    %5 = shapex.make_ranked_shape %1, %4 : (index, index) -> !shapex.ranked_shape<[?,?]>
    %6 = shapex.tie_shape %arg0, %5 : memref<?x?xf32>, !shapex.ranked_shape<[?,?]>
    return
}
// CHECK: %[[DIM0:.+]] = llvm.sext
// CHECK: %[[DIM1:.+]] = llvm.sext
// CHECK: llvm.insertvalue %[[DIM0]], %[[MEMREFBASEPTR:.+]][3, 0]
// CHECK: %[[MEMREFBASEPTR_1:.+]] = llvm.insertvalue %[[DIM1]], %[[MEMREFBASEPTR:.+]][3, 1]
// CHECK: %[[STRIDE1:.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK: %[[MEMREFBASEPTR_2:.+]] = llvm.insertvalue %[[STRIDE1]], %[[MEMREFBASEPTR_1]][4, 1]
// CHECK: %[[ESTRIDE1:.+]] = llvm.extractvalue %[[MEMREFBASEPTR_2]][4, 1] 
// CHECK: %[[EDIM1:.+]] = llvm.extractvalue %[[MEMREFBASEPTR_2]][3, 1] 
// CHECK: %[[STRIDE0:.+]] = llvm.mul %[[ESTRIDE1]], %[[EDIM1]] : !llvm.i64
// CHECK: llvm.insertvalue %[[STRIDE0]], %[[MEMREFBASEPTR_2]][4, 0]