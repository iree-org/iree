// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-decompose-winograd),cse)" --split-input-file %s | FileCheck %s

module {
  func.func @winograd_filter_transform(%arg0: tensor<3x3x64x128xf32>, %arg1: tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32> {
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [3, 3, 1, 1] [1, 1, 1, 1] : tensor<3x3x64x128xf32> to tensor<3x3x1x1xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, 0, 0] [8, 8, 1, 1] [1, 1, 1, 1] : tensor<8x8x64x128xf32> to tensor<8x8x1x1xf32>
    %14 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([0, 1]) ins(%extracted_slice : tensor<3x3x1x1xf32>) outs(%extracted_slice_0 : tensor<8x8x1x1xf32>) -> tensor<8x8x1x1xf32>
    %inserted_slice = tensor.insert_slice %14 into %arg1[0, 0, 0, 0] [8, 8, 1, 1] [1, 1, 1, 1] : tensor<8x8x1x1xf32> into tensor<8x8x64x128xf32>
    return %inserted_slice : tensor<8x8x64x128xf32>
  }
}
// CHECK:      func.func @winograd_filter_transform(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<3x3x64x128xf32>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<8x8x64x128xf32>
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[GT:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, -0.222222224,{{.*}} : tensor<3x8xf32>
// CHECK-DAG:    %[[G:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00,{{.*}} : tensor<8x3xf32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<3x8xf32>
// CHECK-DAG:    %[[INPUT_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-DAG:    %[[OUTPUT_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK:        %[[FILL_0:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<3x8xf32>) -> tensor<3x8xf32>
// CHECK:        %[[MATMUL_0:.+]] = linalg.matmul ins(%[[INPUT_TILE]], %[[GT]]
// CHECK-SAME:     outs(%[[FILL_0]]
// CHECK:        %[[FILL_1:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[OUTPUT_TILE]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:        %[[MATMUL_1:.+]] = linalg.matmul ins(%[[G]], %[[MATMUL_0]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[INSERTED_SLICE_1:.+]] = tensor.insert_slice %[[MATMUL_1]] into %[[ARG1]]
// CHECK:        return %[[INSERTED_SLICE_1]] : tensor<8x8x64x128xf32>

// -----

module {
  func.func @winograd_filter_transform_fchw(%arg0: tensor<64x128x3x3xf32>, %arg1: tensor<8x8x64x128xf32>) -> tensor<8x8x64x128xf32> {
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<64x128x3x3xf32> to tensor<1x1x3x3xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, 0, 0] [8, 8, 1, 1] [1, 1, 1, 1] : tensor<8x8x64x128xf32> to tensor<8x8x1x1xf32>
    %14 = iree_linalg_ext.winograd.filter_transform output_tile_size(6) kernel_size(3) kernel_dimensions([2, 3]) ins(%extracted_slice : tensor<1x1x3x3xf32>) outs(%extracted_slice_0 : tensor<8x8x1x1xf32>) -> tensor<8x8x1x1xf32>
    %inserted_slice = tensor.insert_slice %14 into %arg1[0, 0, 0, 0] [8, 8, 1, 1] [1, 1, 1, 1] : tensor<8x8x1x1xf32> into tensor<8x8x64x128xf32>
    return %inserted_slice : tensor<8x8x64x128xf32>
  }
}
// CHECK:      func.func @winograd_filter_transform_fchw(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<64x128x3x3xf32>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<8x8x64x128xf32>
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:    %[[GT:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, -0.222222224,{{.*}} : tensor<3x8xf32>
// CHECK-DAG:    %[[G:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00,{{.*}} : tensor<8x3xf32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<3x8xf32>
// CHECK-DAG:    %[[INPUT_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-DAG:    %[[OUTPUT_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK:        %[[FILL_0:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<3x8xf32>) -> tensor<3x8xf32>
// CHECK:        %[[MATMUL_0:.+]] = linalg.matmul ins(%[[INPUT_TILE]], %[[GT]]
// CHECK-SAME:     outs(%[[FILL_0]]
// CHECK:        %[[FILL_1:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[OUTPUT_TILE]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:        %[[MATMUL_1:.+]] = linalg.matmul ins(%[[G]], %[[MATMUL_0]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[INSERTED_SLICE_1:.+]] = tensor.insert_slice %[[MATMUL_1]] into %[[ARG1]]
// CHECK:        return %[[INSERTED_SLICE_1]] : tensor<8x8x64x128xf32>

// -----

module {
  func.func @winograd_input_transform(%arg0: tensor<2x130x130x64xf16>, %arg1: tensor<8x8x2x22x22x64xf16>,
                                      %s0 : index, %s1 : index,
                                      %i0 : index, %i1 : index, %i2 : index, %i3 : index, %i4 : index, %i5 : index) -> tensor<8x8x2x22x22x64xf16> {
    %extracted_slice = tensor.extract_slice %arg0[%i0, %i2, %i3, %i1] [1, %s0, %s1, 1] [1, 1, 1, 1] : tensor<2x130x130x64xf16> to tensor<1x?x?x1xf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %i0, %i4, %i5, %i1] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x2x22x22x64xf16> to tensor<8x8x1x1x1x1xf16>
    %14 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<1x?x?x1xf16>) outs(%extracted_slice_0 : tensor<8x8x1x1x1x1xf16>) -> tensor<8x8x1x1x1x1xf16>
    %inserted_slice = tensor.insert_slice %14 into %arg1[0, 0, %i0, %i4, %i5, %i1] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x1x1x1xf16> into tensor<8x8x2x22x22x64xf16>
    return %inserted_slice : tensor<8x8x2x22x22x64xf16>
  }
}
// CHECK:      func.func @winograd_input_transform(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x130x130x64xf16>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<8x8x2x22x22x64xf16>
// CHECK-SAME:   %[[S0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S1:[a-zA-Z0-9_]+]]: index
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:    %[[BT:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00,{{.*}} : tensor<8x8xf32>
// CHECK-DAG:    %[[B:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, -5.250000e+00,{{.*}} : tensor<8x8xf32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<8x8xf16>
// CHECK-DAG:    %[[INPUT_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-DAG:    %[[OUTPUT_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK:        %[[FILL_0:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY]] : tensor<8x8xf16>) -> tensor<8x8xf16>
// CHECK:        %[[INSERTED_SLICE_0:.+]] = tensor.insert_slice %[[INPUT_TILE]] into %[[FILL_0]][0, 0]
// CHECK-SAME:                 [%[[S0]], %[[S1]]] [1, 1] : tensor<?x?xf16> into tensor<8x8xf16>
// CHECK:        %[[FILL_1:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[OUTPUT_TILE]] : tensor<8x8xf16>) -> tensor<8x8xf16>
// CHECK:        %[[MATMUL_0:.+]] = linalg.matmul ins(%[[INSERTED_SLICE_0]], %[[BT]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[MATMUL_1:.+]] = linalg.matmul ins(%[[B]], %[[MATMUL_0]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[INSERTED_SLICE_1:.+]] = tensor.insert_slice %[[MATMUL_1]] into %[[ARG1]]
// CHECK:        return %[[INSERTED_SLICE_1]] : tensor<8x8x2x22x22x64xf16>

// -----

module {
  func.func @winograd_input_transform_nchw(%arg0: tensor<2x64x130x130xf16>, %arg1: tensor<8x8x2x22x22x64xf16>,
                                           %s0 : index, %s1 : index,
                                           %i0 : index, %i1 : index, %i2 : index, %i3 : index, %i4 : index, %i5 : index) -> tensor<8x8x2x22x22x64xf16> {
    %extracted_slice = tensor.extract_slice %arg0[%i0, %i1, %i2, %i3] [1, 1, %s0, %s1] [1, 1, 1, 1] : tensor<2x64x130x130xf16> to tensor<1x1x?x?xf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, %i0, %i4, %i5, %i1] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x2x22x22x64xf16> to tensor<8x8x1x1x1x1xf16>
    %14 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3]) ins(%extracted_slice : tensor<1x1x?x?xf16>) outs(%extracted_slice_0 : tensor<8x8x1x1x1x1xf16>) -> tensor<8x8x1x1x1x1xf16>
    %inserted_slice = tensor.insert_slice %14 into %arg1[0, 0, %i0, %i4, %i5, %i1] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x1x1x1xf16> into tensor<8x8x2x22x22x64xf16>
    return %inserted_slice : tensor<8x8x2x22x22x64xf16>
  }
}
// CHECK:      func.func @winograd_input_transform_nchw(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<2x64x130x130xf16>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<8x8x2x22x22x64xf16>
// CHECK-SAME:   %[[S0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[S1:[a-zA-Z0-9_]+]]: index
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:    %[[BT:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, 0.000000e+00,{{.*}} : tensor<8x8xf32>
// CHECK-DAG:    %[[B:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00, -5.250000e+00,{{.*}} : tensor<8x8xf32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<8x8xf16>
// CHECK-DAG:    %[[INPUT_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-DAG:    %[[OUTPUT_TILE:.+]] = tensor.extract_slice %[[ARG1]]
// CHECK:        %[[FILL_0:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY]] : tensor<8x8xf16>) -> tensor<8x8xf16>
// CHECK:        %[[INSERTED_SLICE_0:.+]] = tensor.insert_slice %[[INPUT_TILE]] into %[[FILL_0]][0, 0]
// CHECK-SAME:                 [%[[S0]], %[[S1]]] [1, 1] : tensor<?x?xf16> into tensor<8x8xf16>
// CHECK:        %[[FILL_1:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[OUTPUT_TILE]] : tensor<8x8xf16>) -> tensor<8x8xf16>
// CHECK:        %[[MATMUL_0:.+]] = linalg.matmul ins(%[[INSERTED_SLICE_0]], %[[BT]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[MATMUL_1:.+]] = linalg.matmul ins(%[[B]], %[[MATMUL_0]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[INSERTED_SLICE_1:.+]] = tensor.insert_slice %[[MATMUL_1]] into %[[ARG1]]
// CHECK:        return %[[INSERTED_SLICE_1]] : tensor<8x8x2x22x22x64xf16>

// -----

module {
  func.func @winograd_output_transform(%arg0: tensor<8x8x1x6x6x32xf16>, %arg1: tensor<1x36x36x32xf16>,
                                       %s0 : index, %s1 : index,
                                       %i0 : index, %i1 : index, %i2 : index, %i3 : index, %i4 : index) -> tensor<1x36x36x32xf16> {
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %i0, %i1, %i2] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x6x6x32xf16> to tensor<8x8x1x1x1x1xf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %i3, %i4, %i2] [1, %s0, %s1, 1] [1, 1, 1, 1] : tensor<1x36x36x32xf16> to tensor<1x?x?x1xf16>
    %12 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<8x8x1x1x1x1xf16>) outs(%extracted_slice_0 : tensor<1x?x?x1xf16>) -> tensor<1x?x?x1xf16>
    %inserted_slice = tensor.insert_slice %12 into %arg1[0, %i3, %i4, %i2] [1, %s0, %s1, 1] [1, 1, 1, 1] : tensor<1x?x?x1xf16> into tensor<1x36x36x32xf16>
    return %inserted_slice : tensor<1x36x36x32xf16>
  }
}
// CHECK:      func.func @winograd_output_transform(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<8x8x1x6x6x32xf16>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<1x36x36x32xf16>
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:    %[[AT:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 1.000000e+00,{{.*}} : tensor<6x8xf32>
// CHECK-DAG:    %[[A:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00,{{.*}} : tensor<8x6xf32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<8x6xf16>
// CHECK-DAG:    %[[INPUT_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-DAG:    %[[OUTPUT_TILE:.+]] = tensor.extract_slice %[[ARG1]]{{.*}} : tensor<1x36x36x32xf16> to tensor<1x?x?x1xf16>
// CHECK-DAG:    %[[OUTPUT_TILE_REDUCED:.+]] = tensor.extract_slice %[[ARG1]]{{.*}} : tensor<1x36x36x32xf16> to tensor<?x?xf16>
// CHECK:        %[[FILL_0:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY]] : tensor<8x6xf16>) -> tensor<8x6xf16>
// CHECK:        %[[MATMUL_0:.+]] = linalg.matmul ins(%[[INPUT_TILE]], %[[A]]
// CHECK-SAME:     outs(%[[FILL_0]]
// CHECK:        %[[FILL_1:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[OUTPUT_TILE_REDUCED]] : tensor<?x?xf16>) -> tensor<?x?xf16>
// CHECK:        %[[MATMUL_1:.+]] = linalg.matmul ins(%[[AT]], %[[MATMUL_0]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[INSERTED_SLICE_0:.+]] = tensor.insert_slice %[[MATMUL_1]] into %[[OUTPUT_TILE]]
// CHECK:        %[[INSERTED_SLICE_1:.+]] = tensor.insert_slice %[[INSERTED_SLICE_0]] into %[[ARG1]]
// CHECK:        return %[[INSERTED_SLICE_1]] : tensor<1x36x36x32xf16>

// -----

module {
  func.func @winograd_output_transform_nchw(%arg0: tensor<8x8x1x6x6x32xf16>, %arg1: tensor<1x32x36x36xf16>,
                                            %s0 : index, %s1 : index,
                                            %i0 : index, %i1 : index, %i2 : index, %i3 : index, %i4 : index) -> tensor<1x32x36x36xf16> {
    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0, %i0, %i1, %i2] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x6x6x32xf16> to tensor<8x8x1x1x1x1xf16>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %i2, %i3, %i4] [1, 1, %s0, %s1] [1, 1, 1, 1] : tensor<1x32x36x36xf16> to tensor<1x1x?x?xf16>
    %12 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3]) ins(%extracted_slice : tensor<8x8x1x1x1x1xf16>) outs(%extracted_slice_0 : tensor<1x1x?x?xf16>) -> tensor<1x1x?x?xf16>
    %inserted_slice = tensor.insert_slice %12 into %arg1[0, %i2, %i3, %i4] [1, 1, %s0, %s1] [1, 1, 1, 1] : tensor<1x1x?x?xf16> into tensor<1x32x36x36xf16>
    return %inserted_slice : tensor<1x32x36x36xf16>
  }
}
// CHECK:      func.func @winograd_output_transform_nchw(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<8x8x1x6x6x32xf16>
// CHECK-SAME:   %[[ARG1:.+]]: tensor<1x32x36x36xf16>
// CHECK-DAG:    %[[ZERO:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:    %[[AT:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 1.000000e+00,{{.*}} : tensor<6x8xf32>
// CHECK-DAG:    %[[A:.+]] = arith.constant dense<{{\[\[}}1.000000e+00, 0.000000e+00,{{.*}} : tensor<8x6xf32>
// CHECK-DAG:    %[[EMPTY:.+]] = tensor.empty() : tensor<8x6xf16>
// CHECK-DAG:    %[[INPUT_TILE:.+]] = tensor.extract_slice %[[ARG0]]
// CHECK-DAG:    %[[OUTPUT_TILE:.+]] = tensor.extract_slice %[[ARG1]]{{.*}} : tensor<1x32x36x36xf16> to tensor<1x1x?x?xf16>
// CHECK-DAG:    %[[OUTPUT_TILE_REDUCED:.+]] = tensor.extract_slice %[[ARG1]]{{.*}} : tensor<1x32x36x36xf16> to tensor<?x?xf16>
// CHECK:        %[[FILL_0:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY]] : tensor<8x6xf16>) -> tensor<8x6xf16>
// CHECK:        %[[MATMUL_0:.+]] = linalg.matmul ins(%[[INPUT_TILE]], %[[A]]
// CHECK-SAME:     outs(%[[FILL_0]]
// CHECK:        %[[FILL_1:.+]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[OUTPUT_TILE_REDUCED]] : tensor<?x?xf16>) -> tensor<?x?xf16>
// CHECK:        %[[MATMUL_1:.+]] = linalg.matmul ins(%[[AT]], %[[MATMUL_0]]
// CHECK-SAME:     outs(%[[FILL_1]]
// CHECK:        %[[INSERTED_SLICE_0:.+]] = tensor.insert_slice %[[MATMUL_1]] into %[[OUTPUT_TILE]]
// CHECK:        %[[INSERTED_SLICE_1:.+]] = tensor.insert_slice %[[INSERTED_SLICE_0]] into %[[ARG1]]
// CHECK:        return %[[INSERTED_SLICE_1]] : tensor<1x32x36x36xf16>
