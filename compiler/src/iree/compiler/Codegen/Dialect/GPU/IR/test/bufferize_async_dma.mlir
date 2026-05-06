// RUN: iree-opt %s --one-shot-bufferize="bufferize-function-boundaries" --split-input-file | FileCheck %s

// Test basic tensor -> memref bufferization.
func.func @bufferize_async_dma(%source: tensor<20x64xf16>,
                                %dest: tensor<1x64xf16>,
                                %i: index, %j: index, %c0: index)
    -> tensor<1x64xf16> {
  %0 = iree_gpu.async_dma %source[%i, %j] to %dest[%c0, %c0], vector<1x64xf16>
      : tensor<20x64xf16>, tensor<1x64xf16> -> tensor<1x64xf16>
  return %0 : tensor<1x64xf16>
}

// CHECK-LABEL: func @bufferize_async_dma
//       CHECK:   iree_gpu.async_dma {{.+}} : memref<20x64xf16{{.*}}>, memref<1x64xf16{{.*}}>

// -----

// Test bufferization with in_bounds attribute.
func.func @bufferize_async_dma_in_bounds(%source: tensor<20x64xf16>,
                                          %dest: tensor<1x64xf16>,
                                          %i: index, %j: index, %c0: index)
    -> tensor<1x64xf16> {
  %0 = iree_gpu.async_dma %source[%i, %j] to %dest[%c0, %c0], vector<1x64xf16>
      in_bounds [true, false]
      : tensor<20x64xf16>, tensor<1x64xf16> -> tensor<1x64xf16>
  return %0 : tensor<1x64xf16>
}

// CHECK-LABEL: func @bufferize_async_dma_in_bounds
//       CHECK:   iree_gpu.async_dma {{.+}} in_bounds [true, false] : memref<20x64xf16{{.*}}>, memref<1x64xf16{{.*}}>

// -----

// Test bufferization with gather (vector) source indices.
func.func @bufferize_async_dma_gather(%source: tensor<1024x64xf16>,
                                       %dest: tensor<1x64xf16>,
                                       %indices: vector<1xindex>,
                                       %j: index, %c0: index)
    -> tensor<1x64xf16> {
  %0 = iree_gpu.async_dma %source[%indices, %j] to %dest[%c0, %c0], vector<1x64xf16>
      : tensor<1024x64xf16> [vector<1xindex>, index],
        tensor<1x64xf16> -> tensor<1x64xf16>
  return %0 : tensor<1x64xf16>
}

// CHECK-LABEL: func @bufferize_async_dma_gather
//       CHECK:   iree_gpu.async_dma {{.+}} : memref<1024x64xf16{{.*}}> [vector<1xindex>, index], memref<1x64xf16{{.*}}>
