// RUN: iree-opt -pass-pipeline='builtin.module(func.func(iree-vector-ext-fold-unit-extent-dims,canonicalize,cse))' %s | FileCheck %s

#layout = #iree_vector_ext.nested_layout<
    subgroup_tile = [1, 1, 1, 1],
    batch_tile = [1, 1, 64, 1],
    outer_tile = [1, 1, 1, 1],
    thread_tile = [1, 1, 2, 128],
    element_tile = [1, 1, 1, 8],

    subgroup_strides = [0, 0, 0, 0],
    thread_strides = [0, 0, 128, 1]
>

func.func @dynamic_shape(%arg0: tensor<1x1x128x?xf16>) -> tensor<1x1x128x?xf16> {
    %to_layout = iree_vector_ext.to_layout %arg0 to layout(#layout) : tensor<1x1x128x?xf16>
    return %to_layout : tensor<1x1x128x?xf16>
}

// CHECK-LABEL: func.func @dynamic_shape
// CHECK-DAG: %[[DIM:.+]] = tensor.dim %arg0, %c3 : tensor<1x1x128x?xf16>
// CHECK-DAG: %[[SLICE:.+]] = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 128, %[[DIM]]] [1, 1, 1, 1] : tensor<1x1x128x?xf16> to tensor<128x?xf16>
// CHECK-DAG: %[[LAYOUT:.+]] = iree_vector_ext.to_layout %[[SLICE]]
// CHECK-DAG: %[[EMPTY:.+]] = tensor.empty(%[[DIM]]) : tensor<1x1x128x?xf16>
// CHECK-DAG: %[[INSERT_SLICE:.+]] = tensor.insert_slice %[[LAYOUT]] into %[[EMPTY]][0, 0, 0, 0] [1, 1, 128, %dim] [1, 1, 1, 1] : tensor<128x?xf16> into tensor<1x1x128x?xf16>
