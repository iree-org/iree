// Tests the printing/parsing of the VMLA dialect ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @vmla_copy
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_INDEX_0:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_INDEX_1:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_INDEX_0:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_INDEX_1:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[LENGTH_0:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[LENGTH_1:[a-zA-Z0-9$._-]+]]
func @vmla_copy(%src : !vmla.buffer,
                %src_shape : !shapex.ranked_shape<[64]>,
                %src_index_0 : index,
                %src_index_1 : index,
                %dst : !vmla.buffer,
                %dst_shape : !shapex.ranked_shape<[32]>,
                %dst_index_0 : index,
                %dst_index_1 : index,
                %length_0 : index,
                %length_1 : index) {
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[64]>),
  // CHECK-SAME; src_indices = [%[[SRC_INDEX_0]], %[[SRC_INDEX_1]]],
  // CHECK-SAME: out %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[32]>),
  // CHECK-SAME: dst_indices = [%[[DST_INDEX_0]], %[[DST_INDEX_1]]],
  // CHECK-SAME: lengths = [%[[LENGTH_0]], %[[LENGTH_1]]] : i32
  vmla.copy %src(%src_shape : !shapex.ranked_shape<[64]>),
            src_indices = [%src_index_0, %src_index_1],
            out %dst(%dst_shape : !shapex.ranked_shape<[32]>),
            dst_indices = [%dst_index_0, %dst_index_1],
            lengths = [%length_0, %length_1] : i32
  return
}

// -----

// CHECK-LABEL: @vmla_copy_no_variadic
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_copy_no_variadic(%src : !vmla.buffer,
                %src_shape : !shapex.ranked_shape<[64]>,
                %dst : !vmla.buffer,
                %dst_shape : !shapex.ranked_shape<[32]>) {
  // CHECK:      vmla.copy
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[64]>),
  // CHECK-SAME: out %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[32]>)
  // CHECK-SAME: : i32
  vmla.copy %src(%src_shape : !shapex.ranked_shape<[64]>),
            out %dst(%dst_shape : !shapex.ranked_shape<[32]>) : i32
  return
}

// -----

// CHECK-LABEL: @vmla_transpose
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_transpose(%src : !vmla.buffer,
                     %src_shape : !shapex.ranked_shape<[64,32,32,10]>,
                     %dst : !vmla.buffer,
                     %dst_shape : !shapex.ranked_shape<[64,10,32,32]>) {
  // CHECK:      vmla.transpose
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[64,32,32,10]>),
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[64,10,32,32]>)
  // CHECK-SAME: {permutation = dense<[0, 3, 2, 1]> : tensor<4xi32>} : f32
  vmla.transpose %src(%src_shape : !shapex.ranked_shape<[64,32,32,10]>),
                 out %dst(%dst_shape : !shapex.ranked_shape<[64,10,32,32]>)
                 {permutation = dense<[0, 3, 2, 1]> : tensor<4xi32>} : f32
  return
}

// -----

// CHECK-LABEL: @vmla_reverse
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_reverse(%src : !vmla.buffer,
                   %src_shape : !shapex.ranked_shape<[4,8]>,
                   %dst : !vmla.buffer,
                   %dst_shape : !shapex.ranked_shape<[4,8]>) {
  // CHECK:      vmla.reverse
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[4,8]>)
  // CHECK-SAME: {dimensions = dense<1> : tensor<1xi32>} : f32
  vmla.reverse %src(%src_shape : !shapex.ranked_shape<[4,8]>),
               out %dst(%dst_shape : !shapex.ranked_shape<[4,8]>)
               {dimensions = dense<1> : tensor<1xi32>} : f32
  return
}

// -----

// CHECK-LABEL: @vmla_pad
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[VALUE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[VALUE_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_pad(%src : !vmla.buffer,
               %src_shape : !shapex.ranked_shape<[4,8]>,
               %value : !vmla.buffer,
               %value_shape : !shapex.ranked_shape<[4,8]>,
               %dst : !vmla.buffer,
               %dst_shape : !shapex.ranked_shape<[4,8]>) {
  // CHECK:      vmla.pad
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: %[[VALUE]](%[[VALUE_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[4,8]>)
  // CHECK-SAME: {edge_padding_high = dense<2> : tensor<i32>,
  // CHECK-SAME: edge_padding_low = dense<2> : tensor<i32>,
  // CHECK-SAME: interior_padding = dense<0> : tensor<i32>} : f32
  vmla.pad %src(%src_shape : !shapex.ranked_shape<[4,8]>),
           %value(%value_shape : !shapex.ranked_shape<[4,8]>),
           out %dst(%dst_shape : !shapex.ranked_shape<[4,8]>)
           {edge_padding_high = dense<2> : tensor<i32>,
            edge_padding_low = dense<2> : tensor<i32>,
            interior_padding = dense<0> : tensor<i32>} : f32
  return
}

// -----

// CHECK-LABEL: @vmla_broadcast
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_broadcast(%src : !vmla.buffer,
                     %src_shape : !shapex.ranked_shape<[]>,
                     %dst : !vmla.buffer,
                     %dst_shape : !shapex.ranked_shape<[4,8]>) {
  // CHECK:      vmla.broadcast
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[]>),
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[4,8]>) : f32
  vmla.broadcast %src(%src_shape : !shapex.ranked_shape<[]>),
                 out %dst(%dst_shape : !shapex.ranked_shape<[4,8]>) : f32
  return
}

// -----

// CHECK-LABEL: @vmla_tile
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_tile(%src : !vmla.buffer,
                %src_shape : !shapex.ranked_shape<[4]>,
                %dst : !vmla.buffer,
                %dst_shape : !shapex.ranked_shape<[4,8]>) {
  // CHECK:      vmla.tile
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[4]>),
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[4,8]>) : f32
  vmla.tile %src(%src_shape : !shapex.ranked_shape<[4]>),
            out %dst(%dst_shape : !shapex.ranked_shape<[4,8]>) : f32
  return
}

// -----

// CHECK-LABEL: @vmla_gather
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[INDICES:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[INDICES_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_gather(%src : !vmla.buffer,
                  %src_shape : !shapex.ranked_shape<[4,8]>,
                  %indices : !vmla.buffer,
                  %indices_shape : !shapex.ranked_shape<[4,8]>,
                  %dst : !vmla.buffer,
                  %dst_shape : !shapex.ranked_shape<[4,8]>) {
  // CHECK:      vmla.gather
  // CHECK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: %[[INDICES]](%[[INDICES_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: out
  // CHECK-SAME: %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[4,8]>)
  // CHECK-SAME: {batch_dims = 2 : i64, dim = 1 : i64} : f32
  vmla.gather %src(%src_shape : !shapex.ranked_shape<[4,8]>),
              %indices(%indices_shape : !shapex.ranked_shape<[4,8]>),
              out %dst(%dst_shape : !shapex.ranked_shape<[4,8]>)
              {batch_dims = 2 : i64, dim = 1 : i64} : f32
  return
}
