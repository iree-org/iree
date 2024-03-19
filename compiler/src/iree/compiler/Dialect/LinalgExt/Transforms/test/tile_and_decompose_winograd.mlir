// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-tile-and-decompose-winograd),cse)" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-linalg-ext-tile-and-decompose-winograd{onlyTile}),cse)" --split-input-file %s | FileCheck %s --check-prefix=TILING

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_input_transform(%arg0: tensor<1x10x10x1280xf32>) -> tensor<8x8x1x2x2x1280xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1280 = arith.constant 1280 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<8x8x1x2x2x1280xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c1280 step %c32 iter_args(%arg4 = %arg2) -> (tensor<8x8x1x2x2x1280xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c1280]
        %extracted_slice = tensor.extract_slice %arg0[%arg1, 0, 0, %arg3] [%2, 10, 10, %4] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
        %extracted_slice_0 = tensor.extract_slice %0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to tensor<8x8x?x2x2x?xf32>
        %5 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<?x10x10x?xf32>) outs(%extracted_slice_0 : tensor<8x8x?x2x2x?xf32>) -> tensor<8x8x?x2x2x?xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into tensor<8x8x1x2x2x1280xf32>
        scf.yield %inserted_slice : tensor<8x8x1x2x2x1280xf32>
      }
      scf.yield %3 : tensor<8x8x1x2x2x1280xf32>
    }
    return %1 : tensor<8x8x1x2x2x1280xf32>
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// CHECK:      func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8xf32>
// CHECK:        %[[CST:.+]] = arith.constant dense<
// CHECK:        %[[CST_0:.+]] = arith.constant dense<
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:        %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 10,
// CHECK-SAME:         10, %[[D5]]] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// CHECK-SAME:         tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// CHECK-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// CHECK:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// CHECK-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// CHECK:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// CHECK-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]], %[[D8]],
// CHECK-SAME:                 %[[D11]], %[[ARG11]]] [1, %[[D9]], %[[D12]], 1] [1, 1, 1, 1] : tensor<?x10x10x?xf32> to
// CHECK-SAME:                 tensor<?x?xf32>
// CHECK:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// CHECK-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// CHECK-SAME:                 tensor<8x8xf32>
// CHECK:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x8xf32>) ->
// CHECK-SAME:                 tensor<8x8xf32>
// CHECK:                    %[[INSERTED_SLICE_4:.+]] = tensor.insert_slice %[[EXTRACTED_SLICE_3]] into %[[D14]][0, 0]
// CHECK-SAME:                 [%[[D9]], %[[D12]]] [1, 1] : tensor<?x?xf32> into tensor<8x8xf32>
// CHECK:                    %[[D15:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_5]] :
// CHECK-SAME:                 tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                    %[[D16:.+]] = linalg.matmul ins(%[[INSERTED_SLICE_4]], %[[CST_0]] : tensor<8x8xf32>,
// CHECK-SAME:                 tensor<8x8xf32>) outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                    %[[D18:.+]] = linalg.matmul ins(%[[CST]], %[[D16]] : tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK-SAME:                 outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[D18]] into %[[ARG12]][0, 0, %[[ARG5]],
// CHECK-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// CHECK-SAME:                 into tensor<8x8x?x2x2x?xf32>
// CHECK:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                  }
// CHECK:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                }
// CHECK:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// CHECK:              }
// CHECK:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// CHECK:            }
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// CHECK-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// CHECK-SAME:         tensor<8x8x1x2x2x1280xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:        }
// CHECK:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }

// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// TILING:      func.func @winograd_input_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x10x10x1280xf32>) ->
// TILING-SAME:   tensor<8x8x1x2x2x1280xf32> {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C1280:.+]] = arith.constant 1280 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// TILING-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// TILING-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 10,
// TILING-SAME:         10, %[[D5]]] [1, 1, 1, 1] : tensor<1x10x10x1280xf32> to tensor<?x10x10x?xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// TILING-SAME:         tensor<8x8x?x2x2x?xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// TILING-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// TILING:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// TILING:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// TILING-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]], %[[D8]],
// TILING-SAME:                 %[[D11]], %[[ARG11]]] [1, %[[D9]], %[[D12]], 1] [1, 1, 1, 1] : tensor<?x10x10x?xf32> to
// TILING-SAME:                 tensor<?x?xf32>
// TILING:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// TILING-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// TILING-SAME:                 tensor<8x8xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<?x?xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_5]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// TILING:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// TILING-SAME:                 into tensor<8x8x?x2x2x?xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// TILING:                }
// TILING:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// TILING-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// TILING-SAME:         tensor<8x8x1x2x2x1280xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// TILING:      }

// -----

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_output_transform(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x12x12x32xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<1x12x12x32xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<1x12x12x32xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c32 step %c32 iter_args(%arg4 = %arg2) -> (tensor<1x12x12x32xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c32]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
        %extracted_slice_0 = tensor.extract_slice %0[%arg1, 0, 0, %arg3] [%2, 12, 12, %4] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<?x12x12x?xf32>
        %5 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2]) ins(%extracted_slice : tensor<8x8x?x2x2x?xf32>) outs(%extracted_slice_0 : tensor<?x12x12x?xf32>) -> tensor<?x12x12x?xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg1, 0, 0, %arg3] [%2, 12, 12, %4] [1, 1, 1, 1] : tensor<?x12x12x?xf32> into tensor<1x12x12x32xf32>
        scf.yield %inserted_slice : tensor<1x12x12x32xf32>
      }
      scf.yield %3 : tensor<1x12x12x32xf32>
    }
    return %1 : tensor<1x12x12x32xf32>
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// CHECK-SAME:   tensor<1x12x12x32xf32> {
// CHECK:        %[[CST:.+]] = arith.constant dense<
// CHECK:        %[[CST_0:.+]] = arith.constant dense<
// CHECK:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x6xf32>
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<1x12x12x32xf32>
// CHECK:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<1x12x12x32xf32>) {
// CHECK-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x12x12x32xf32>) {
// CHECK-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 12,
// CHECK-SAME:         12, %[[D5]]] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<?x12x12x?xf32>
// CHECK:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<?x12x12x?xf32>) {
// CHECK:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<?x12x12x?xf32>) {
// CHECK-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// CHECK:                %[[D9:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<?x12x12x?xf32>) {
// CHECK-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// CHECK:                  %[[D11:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// CHECK-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<?x12x12x?xf32>) {
// CHECK:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// CHECK-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// CHECK-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// CHECK:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[D8]], %[[D10]],
// CHECK-SAME:                 %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<?x12x12x?xf32> to tensor<6x6xf32>
// CHECK:                    %[[D12:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x6xf32>) ->
// CHECK-SAME:                 tensor<8x6xf32>
// CHECK:                    %[[D13:.+]] = linalg.matmul ins(%[[EXTRACTED_SLICE_3]], %[[CST_0]] : tensor<8x8xf32>,
// CHECK-SAME:                 tensor<8x6xf32>) outs(%[[D12]] : tensor<8x6xf32>) -> tensor<8x6xf32>
// CHECK:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_4]] :
// CHECK-SAME:                 tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:                    %[[D15:.+]] = linalg.matmul ins(%[[CST]], %[[D13]] : tensor<6x8xf32>, tensor<8x6xf32>)
// CHECK-SAME:                 outs(%[[D14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[D15]] into %[[ARG12]][%[[ARG5]], %[[D8]],
// CHECK-SAME:                 %[[D10]], %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<6x6xf32> into
// CHECK-SAME:                 tensor<?x12x12x?xf32>
// CHECK:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x12x12x?xf32>
// CHECK:                  }
// CHECK:                  scf.yield %[[D11]] : tensor<?x12x12x?xf32>
// CHECK:                }
// CHECK:                scf.yield %[[D9]] : tensor<?x12x12x?xf32>
// CHECK:              }
// CHECK:              scf.yield %[[D7]] : tensor<?x12x12x?xf32>
// CHECK:            }
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], 0, 0, %[[ARG3]]]
// CHECK-SAME:         [%[[D3]], 12, 12, %[[D5]]] [1, 1, 1, 1] : tensor<?x12x12x?xf32> into tensor<1x12x12x32xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<1x12x12x32xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D4]] : tensor<1x12x12x32xf32>
// CHECK:        }
// CHECK:        return %[[D2]] : tensor<1x12x12x32xf32>
// CHECK:      }

// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING:      func.func @winograd_output_transform(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<8x8x1x2x2x32xf32>) ->
// TILING-SAME:   tensor<1x12x12x32xf32> {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<1x12x12x32xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// TILING-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<1x12x12x32xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C32]] step %[[C32]]
// TILING-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<1x12x12x32xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], 0, 0, %[[ARG3]]] [%[[D3]], 12,
// TILING-SAME:         12, %[[D5]]] [1, 1, 1, 1] : tensor<1x12x12x32xf32> to tensor<?x12x12x?xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// TILING-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<?x12x12x?xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<?x12x12x?xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING:                %[[D9:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<?x12x12x?xf32>) {
// TILING-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING:                  %[[D11:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// TILING-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<?x12x12x?xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// TILING-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// TILING:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[D8]], %[[D10]],
// TILING-SAME:                 %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<?x12x12x?xf32> to tensor<6x6xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([1, 2])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<8x8xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_4]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// TILING:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][%[[ARG5]], %[[D8]],
// TILING-SAME:                 %[[D10]], %[[ARG11]]] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<6x6xf32> into
// TILING-SAME:                 tensor<?x12x12x?xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x12x12x?xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D11]] : tensor<?x12x12x?xf32>
// TILING:                }
// TILING:                scf.yield %[[D9]] : tensor<?x12x12x?xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<?x12x12x?xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], 0, 0, %[[ARG3]]]
// TILING-SAME:         [%[[D3]], 12, 12, %[[D5]]] [1, 1, 1, 1] : tensor<?x12x12x?xf32> into tensor<1x12x12x32xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<1x12x12x32xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<1x12x12x32xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<1x12x12x32xf32>
// TILING:      }

// -----

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_input_transform_nchw(%arg0: tensor<1x1280x10x10xf32>) -> tensor<8x8x1x2x2x1280xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1280 = arith.constant 1280 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<8x8x1x2x2x1280xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c1280 step %c32 iter_args(%arg4 = %arg2) -> (tensor<8x8x1x2x2x1280xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c1280]
        %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg3, 0, 0] [%2, %4, 10, 10] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<?x?x10x10xf32>
        %extracted_slice_0 = tensor.extract_slice %0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to tensor<8x8x?x2x2x?xf32>
        %5 = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3]) ins(%extracted_slice : tensor<?x?x10x10xf32>) outs(%extracted_slice_0 : tensor<8x8x?x2x2x?xf32>) -> tensor<8x8x?x2x2x?xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into tensor<8x8x1x2x2x1280xf32>
        scf.yield %inserted_slice : tensor<8x8x1x2x2x1280xf32>
      }
      scf.yield %3 : tensor<8x8x1x2x2x1280xf32>
    }
    return %1 : tensor<8x8x1x2x2x1280xf32>
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// CHECK:      func.func @winograd_input_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1280x10x10xf32>) ->
// CHECK-SAME:   tensor<8x8x1x2x2x1280xf32> {
// CHECK:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x8xf32>
// CHECK:        %[[CST:.+]] = arith.constant dense<
// CHECK:        %[[CST_0:.+]] = arith.constant dense<
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[C1280:.+]] = arith.constant 1280 : index
// CHECK:        %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// CHECK:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// CHECK-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// CHECK-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// CHECK-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// CHECK-SAME:         %[[D5]], 10, 10] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<?x?x10x10xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// CHECK-SAME:         tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// CHECK-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// CHECK-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// CHECK:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// CHECK-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// CHECK:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// CHECK-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// CHECK:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]],
// CHECK-SAME:                 %[[ARG11]], %[[D8]], %[[D11]]] [1, 1, %[[D9]], %[[D12]]] [1, 1, 1, 1] :
// CHECK-SAME:                 tensor<?x?x10x10xf32> to tensor<?x?xf32>
// CHECK:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// CHECK-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// CHECK-SAME:                 tensor<8x8xf32>
// CHECK:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x8xf32>) ->
// CHECK-SAME:                 tensor<8x8xf32>
// CHECK:                    %[[INSERTED_SLICE_4:.+]] = tensor.insert_slice %[[EXTRACTED_SLICE_3]] into %[[D14]][0, 0]
// CHECK-SAME:                 [%[[D9]], %[[D12]]] [1, 1] : tensor<?x?xf32> into tensor<8x8xf32>
// CHECK:                    %[[D15:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_5]] :
// CHECK-SAME:                 tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                    %[[D16:.+]] = linalg.matmul ins(%[[INSERTED_SLICE_4]], %[[CST_0]] : tensor<8x8xf32>,
// CHECK-SAME:                 tensor<8x8xf32>) outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                    %[[D18:.+]] = linalg.matmul ins(%[[CST]], %[[D16]] : tensor<8x8xf32>, tensor<8x8xf32>)
// CHECK-SAME:                 outs(%[[D15]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[D18]] into %[[ARG12]][0, 0, %[[ARG5]],
// CHECK-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// CHECK-SAME:                 into tensor<8x8x?x2x2x?xf32>
// CHECK:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                  }
// CHECK:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// CHECK:                }
// CHECK:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// CHECK:              }
// CHECK:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// CHECK:            }
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// CHECK-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// CHECK-SAME:         tensor<8x8x1x2x2x1280xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:        }
// CHECK:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// CHECK:      }
// CHECK:    }

// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING-DAG:  #[[MAP3:.+]] = affine_map<(d0) -> (-d0 + 10, 8)>
// TILING:      func.func @winograd_input_transform_nchw(%[[ARG0:[a-zA-Z0-9_]+]]: tensor<1x1280x10x10xf32>) ->
// TILING-SAME:   tensor<8x8x1x2x2x1280xf32> {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C1280:.+]] = arith.constant 1280 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<8x8x1x2x2x1280xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1]] step %[[C1]]
// TILING-SAME:     iter_args(%[[ARG2:[a-zA-Z0-9_]+]] = %[[D1]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1280]] step %[[C32]]
// TILING-SAME:       iter_args(%[[ARG4:[a-zA-Z0-9_]+]] = %[[ARG2]]) -> (tensor<8x8x1x2x2x1280xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C1280]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// TILING-SAME:         %[[D5]], 10, 10] [1, 1, 1, 1] : tensor<1x1280x10x10xf32> to tensor<?x?x10x10xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x1280xf32> to
// TILING-SAME:         tensor<8x8x?x2x2x?xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D3]] step %[[C1]]
// TILING-SAME:         iter_args(%[[ARG6:[a-zA-Z0-9_]+]] = %[[EXTRACTED_SLICE_2]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:           iter_args(%[[ARG8:[a-zA-Z0-9_]+]] = %[[ARG6]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING-DAG:              %[[D9:.+]] = affine.min #[[MAP3]](%[[D8]])
// TILING:                %[[D10:.+]] = scf.for %[[ARG9:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C2]] step %[[C1]]
// TILING-SAME:             iter_args(%[[ARG10:[a-zA-Z0-9_]+]] = %[[ARG8]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING-DAG:                %[[D11:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING-DAG:                %[[D12:.+]] = affine.min #[[MAP3]](%[[D11]])
// TILING:                  %[[D13:.+]] = scf.for %[[ARG11:[a-zA-Z0-9_]+]] = %[[C0]] to %[[D5]] step %[[C1]]
// TILING-SAME:               iter_args(%[[ARG12:[a-zA-Z0-9_]+]] = %[[ARG10]]) -> (tensor<8x8x?x2x2x?xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][%[[ARG5]],
// TILING-SAME:                 %[[ARG11]], %[[D8]], %[[D11]]] [1, 1, %[[D9]], %[[D12]]] [1, 1, 1, 1] :
// TILING-SAME:                 tensor<?x?x10x10xf32> to tensor<?x?xf32>
// TILING:                    %[[EXTRACTED_SLICE_5:.+]] = tensor.extract_slice %[[ARG12]][0, 0, %[[ARG5]], %[[ARG7]],
// TILING-SAME:                 %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> to
// TILING-SAME:                 tensor<8x8xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.input_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<?x?xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_5]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// TILING:                    %[[INSERTED_SLICE_6:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] : tensor<8x8xf32>
// TILING-SAME:                 into tensor<8x8x?x2x2x?xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_6]] : tensor<8x8x?x2x2x?xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D13]] : tensor<8x8x?x2x2x?xf32>
// TILING:                }
// TILING:                scf.yield %[[D10]] : tensor<8x8x?x2x2x?xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<8x8x?x2x2x?xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][0, 0, %[[ARG1]], 0, 0,
// TILING-SAME:         %[[ARG3]]] [8, 8, %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x?x2x2x?xf32> into
// TILING-SAME:         tensor<8x8x1x2x2x1280xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<8x8x1x2x2x1280xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<8x8x1x2x2x1280xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<8x8x1x2x2x1280xf32>
// TILING:      }
// TILING:    }

// -----

#map = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
#map1 = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
module {
  func.func @winograd_output_transform_nchw(%arg0: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = tensor.empty() : tensor<1x32x12x12xf32>
    %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %0) -> (tensor<1x32x12x12xf32>) {
      %2 = affine.min #map(%arg1)[%c1, %c1]
      %3 = scf.for %arg3 = %c0 to %c32 step %c32 iter_args(%arg4 = %arg2) -> (tensor<1x32x12x12xf32>) {
        %4 = affine.min #map1(%arg3)[%c32, %c32]
        %extracted_slice = tensor.extract_slice %arg0[0, 0, %arg1, 0, 0, %arg3] [8, 8, %2, 2, 2, %4] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
        %extracted_slice_0 = tensor.extract_slice %0[%arg1, %arg3, 0, 0] [%2, %4, 12, 12] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<?x?x12x12xf32>
        %5 = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3]) ins(%extracted_slice : tensor<8x8x?x2x2x?xf32>) outs(%extracted_slice_0 : tensor<?x?x12x12xf32>) -> tensor<?x?x12x12xf32>
        %inserted_slice = tensor.insert_slice %5 into %arg4[%arg1, %arg3, 0, 0] [%2, %4, 12, 12] [1, 1, 1, 1] : tensor<?x?x12x12xf32> into tensor<1x32x12x12xf32>
        scf.yield %inserted_slice : tensor<1x32x12x12xf32>
      }
      scf.yield %3 : tensor<1x32x12x12xf32>
    }
    return %1 : tensor<1x32x12x12xf32>
  }
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// CHECK:      func.func @winograd_output_transform_nchw(%[[ARG0:.+]]: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32>
// CHECK-SAME:   {
// CHECK:        %[[CST:.+]] = arith.constant dense<
// CHECK:        %[[CST_0:.+]] = arith.constant dense<
// CHECK:        %[[CST_1:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = tensor.empty() : tensor<8x6xf32>
// CHECK:        %[[C0:.+]] = arith.constant 0 : index
// CHECK:        %[[C1:.+]] = arith.constant 1 : index
// CHECK:        %[[C2:.+]] = arith.constant 2 : index
// CHECK:        %[[C32:.+]] = arith.constant 32 : index
// CHECK:        %[[D1:.+]] = tensor.empty() : tensor<1x32x12x12xf32>
// CHECK:        %[[D2:.+]] = scf.for %[[ARG1:.+]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[D1]]) ->
// CHECK-SAME:     (tensor<1x32x12x12xf32>) {
// CHECK-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// CHECK:          %[[D4:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C32]] step %[[C32]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) ->
// CHECK-SAME:       (tensor<1x32x12x12xf32>) {
// CHECK-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// CHECK:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// CHECK-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// CHECK:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// CHECK-SAME:         %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<?x?x12x12xf32>
// CHECK:            %[[D6:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[D3]] step %[[C1]] iter_args(%[[ARG6:.+]] =
// CHECK-SAME:         %[[EXTRACTED_SLICE_2]]) -> (tensor<?x?x12x12xf32>) {
// CHECK:              %[[D7:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) ->
// CHECK-SAME:           (tensor<?x?x12x12xf32>) {
// CHECK-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// CHECK:                %[[D9:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]])
// CHECK-SAME:             -> (tensor<?x?x12x12xf32>) {
// CHECK-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// CHECK:                  %[[D11:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[D5]] step %[[C1]] iter_args(%[[ARG12:.+]] =
// CHECK-SAME:               %[[ARG10]]) -> (tensor<?x?x12x12xf32>) {
// CHECK:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// CHECK-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// CHECK-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// CHECK:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[ARG11]], %[[D8]],
// CHECK-SAME:                 %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<?x?x12x12xf32> to tensor<6x6xf32>
// CHECK:                    %[[D12:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[D0]] : tensor<8x6xf32>) ->
// CHECK-SAME:                 tensor<8x6xf32>
// CHECK:                    %[[D13:.+]] = linalg.matmul ins(%[[EXTRACTED_SLICE_3]], %[[CST_0]] : tensor<8x8xf32>,
// CHECK-SAME:                 tensor<8x6xf32>) outs(%[[D12]] : tensor<8x6xf32>) -> tensor<8x6xf32>
// CHECK:                    %[[D14:.+]] = linalg.fill ins(%[[CST_1]] : f32) outs(%[[EXTRACTED_SLICE_4]] : tensor<6x6xf32>)
// CHECK-SAME:                 -> tensor<6x6xf32>
// CHECK:                    %[[D15:.+]] = linalg.matmul ins(%[[CST]], %[[D13]] : tensor<6x8xf32>, tensor<8x6xf32>)
// CHECK-SAME:                 outs(%[[D14]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// CHECK:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[D15]] into %[[ARG12]][%[[ARG5]],
// CHECK-SAME:                 %[[ARG11]], %[[D8]], %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<6x6xf32> into
// CHECK-SAME:                 tensor<?x?x12x12xf32>
// CHECK:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x?x12x12xf32>
// CHECK:                  }
// CHECK:                  scf.yield %[[D11]] : tensor<?x?x12x12xf32>
// CHECK:                }
// CHECK:                scf.yield %[[D9]] : tensor<?x?x12x12xf32>
// CHECK:              }
// CHECK:              scf.yield %[[D7]] : tensor<?x?x12x12xf32>
// CHECK:            }
// CHECK:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], %[[ARG3]], 0, 0]
// CHECK-SAME:         [%[[D3]], %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<?x?x12x12xf32> into tensor<1x32x12x12xf32>
// CHECK:            scf.yield %[[INSERTED_SLICE]] : tensor<1x32x12x12xf32>
// CHECK:          }
// CHECK:          scf.yield %[[D4]] : tensor<1x32x12x12xf32>
// CHECK:        }
// CHECK:        return %[[D2]] : tensor<1x32x12x12xf32>
// CHECK:      }

// TILING-DAG:  #[[MAP:.+]] = affine_map<(d0)[s0, s1] -> (1, -d0 + s1)>
// TILING-DAG:  #[[MAP1:.+]] = affine_map<(d0)[s0, s1] -> (32, -d0 + s1)>
// TILING-DAG:  #[[MAP2:.+]] = affine_map<(d0) -> (d0 * 6)>
// TILING:      func.func @winograd_output_transform_nchw(%[[ARG0:.+]]: tensor<8x8x1x2x2x32xf32>) -> tensor<1x32x12x12xf32>
// TILING-SAME:   {
// TILING:        %[[C0:.+]] = arith.constant 0 : index
// TILING:        %[[C1:.+]] = arith.constant 1 : index
// TILING:        %[[C2:.+]] = arith.constant 2 : index
// TILING:        %[[C32:.+]] = arith.constant 32 : index
// TILING:        %[[D1:.+]] = tensor.empty() : tensor<1x32x12x12xf32>
// TILING:        %[[D2:.+]] = scf.for %[[ARG1:.+]] = %[[C0]] to %[[C1]] step %[[C1]] iter_args(%[[ARG2:.+]] = %[[D1]]) ->
// TILING-SAME:     (tensor<1x32x12x12xf32>) {
// TILING-DAG:        %[[D3:.+]] = affine.min #[[MAP]](%[[ARG1]])[%[[C1]], %[[C1]]]
// TILING:          %[[D4:.+]] = scf.for %[[ARG3:.+]] = %[[C0]] to %[[C32]] step %[[C32]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) ->
// TILING-SAME:       (tensor<1x32x12x12xf32>) {
// TILING-DAG:          %[[D5:.+]] = affine.min #[[MAP1]](%[[ARG3]])[%[[C32]], %[[C32]]]
// TILING:            %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, 0, %[[ARG1]], 0, 0, %[[ARG3]]] [8, 8,
// TILING-SAME:         %[[D3]], 2, 2, %[[D5]]] [1, 1, 1, 1, 1, 1] : tensor<8x8x1x2x2x32xf32> to tensor<8x8x?x2x2x?xf32>
// TILING:            %[[EXTRACTED_SLICE_2:.+]] = tensor.extract_slice %[[D1]][%[[ARG1]], %[[ARG3]], 0, 0] [%[[D3]],
// TILING-SAME:         %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<1x32x12x12xf32> to tensor<?x?x12x12xf32>
// TILING:            %[[D6:.+]] = scf.for %[[ARG5:.+]] = %[[C0]] to %[[D3]] step %[[C1]] iter_args(%[[ARG6:.+]] =
// TILING-SAME:         %[[EXTRACTED_SLICE_2]]) -> (tensor<?x?x12x12xf32>) {
// TILING:              %[[D7:.+]] = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) ->
// TILING-SAME:           (tensor<?x?x12x12xf32>) {
// TILING-DAG:              %[[D8:.+]] = affine.apply #[[MAP2]](%[[ARG7]])
// TILING:                %[[D9:.+]] = scf.for %[[ARG9:.+]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ARG10:.+]] = %[[ARG8]])
// TILING-SAME:             -> (tensor<?x?x12x12xf32>) {
// TILING-DAG:                %[[D10:.+]] = affine.apply #[[MAP2]](%[[ARG9]])
// TILING:                  %[[D11:.+]] = scf.for %[[ARG11:.+]] = %[[C0]] to %[[D5]] step %[[C1]] iter_args(%[[ARG12:.+]] =
// TILING-SAME:               %[[ARG10]]) -> (tensor<?x?x12x12xf32>) {
// TILING:                    %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[EXTRACTED_SLICE]][0, 0, %[[ARG5]],
// TILING-SAME:                 %[[ARG7]], %[[ARG9]], %[[ARG11]]] [8, 8, 1, 1, 1, 1] [1, 1, 1, 1, 1, 1] :
// TILING-SAME:                 tensor<8x8x?x2x2x?xf32> to tensor<8x8xf32>
// TILING:                    %[[EXTRACTED_SLICE_4:.+]] = tensor.extract_slice %[[ARG12]][%[[ARG5]], %[[ARG11]], %[[D8]],
// TILING-SAME:                 %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<?x?x12x12xf32> to tensor<6x6xf32>
// TILING:                    %[[TILED_WINOGRAD:.+]] = iree_linalg_ext.winograd.output_transform output_tile_size(6) kernel_size(3) image_dimensions([2, 3])
// TILING-SAME:                                                  ins(%[[EXTRACTED_SLICE_3]] : tensor<8x8xf32>)
// TILING-SAME:                                                  outs(%[[EXTRACTED_SLICE_4]] : tensor<6x6xf32>) -> tensor<6x6xf32>
// TILING:                    %[[INSERTED_SLICE_5:.+]] = tensor.insert_slice %[[TILED_WINOGRAD]] into %[[ARG12]][%[[ARG5]],
// TILING-SAME:                 %[[ARG11]], %[[D8]], %[[D10]]] [1, 1, 6, 6] [1, 1, 1, 1] : tensor<6x6xf32> into
// TILING-SAME:                 tensor<?x?x12x12xf32>
// TILING:                    scf.yield %[[INSERTED_SLICE_5]] : tensor<?x?x12x12xf32>
// TILING:                  }
// TILING:                  scf.yield %[[D11]] : tensor<?x?x12x12xf32>
// TILING:                }
// TILING:                scf.yield %[[D9]] : tensor<?x?x12x12xf32>
// TILING:              }
// TILING:              scf.yield %[[D7]] : tensor<?x?x12x12xf32>
// TILING:            }
// TILING:            %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[D6]] into %[[ARG4]][%[[ARG1]], %[[ARG3]], 0, 0]
// TILING-SAME:         [%[[D3]], %[[D5]], 12, 12] [1, 1, 1, 1] : tensor<?x?x12x12xf32> into tensor<1x32x12x12xf32>
// TILING:            scf.yield %[[INSERTED_SLICE]] : tensor<1x32x12x12xf32>
// TILING:          }
// TILING:          scf.yield %[[D4]] : tensor<1x32x12x12xf32>
// TILING:        }
// TILING:        return %[[D2]] : tensor<1x32x12x12xf32>
// TILING:      }

