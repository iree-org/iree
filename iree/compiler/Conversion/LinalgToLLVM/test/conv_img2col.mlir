// RUN: iree-opt -iree-codegen-convert-to-llvm -split-input-file --iree-codegen-linalg-to-llvm-conv-img2col-conversion-pass %s | IreeFileCheck %s

func @conv_16433136(%arg0: memref<1x16x16x4xf32>, %arg1: memref<3x3x4x16xf32>, %arg2: memref<1x14x14x16xf32>) {
    linalg.conv(%arg1, %arg0, %arg2)  : memref<3x3x4x16xf32>, memref<1x16x16x4xf32>, memref<1x14x14x16xf32>
    return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: func @conv_16433136(%[[INPUT:.+]]: memref<1x16x16x4xf32>, %[[FILTER:.+]]: memref<3x3x4x16xf32>, %[[OUTPUT:.+]]: memref<1x14x14x16xf32>)
// CHECK: %[[COLBUFFER:.+]] =   alloca() : memref<1x14x14x3x3x4xf32>
// CHECK: linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[INPUT]] : memref<1x16x16x4xf32>) outs(%[[COLBUFFER]] :  memref<1x14x14x3x3x4xf32>)
// CHECK: %[[X:.+]] = linalg.reshape %[[COLBUFFER]] [#[[MAP2]], #[[MAP3]]] : memref<1x14x14x3x3x4xf32> into memref<196x36xf32>
// CHECK: %[[W:.+]] = linalg.reshape %[[FILTER]] [#[[MAP4]], #[[MAP5]]] : memref<3x3x4x16xf32> into memref<36x16xf32>
// CHECK: %[[Y:.+]] = linalg.reshape %[[OUTPUT]] [#[[MAP4]], #[[MAP5]]] : memref<1x14x14x16xf32> into memref<196x16xf32>
// CHECK: linalg.matmul ins(%[[X]], %[[W]] : memref<196x36xf32>, memref<36x16xf32>) outs(%[[Y]] : memref<196x16xf32>
