// Tests the printing/parsing of the VMLA dialect ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @vmla_conv
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[INPUT_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[FILTER:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[FILTER_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_conv(%input : !vmla.buffer,
                %input_shape : !shapex.ranked_shape<[1,4,5,2]>,
                %filter : !vmla.buffer,
                %filter_shape : !shapex.ranked_shape<[3,2,2,1]>,
                %dst : !vmla.buffer,
                %dst_shape : !shapex.ranked_shape<[1,2,3,1]>) {
  // CHECK:      vmla.conv
  // CHECK-SAME: %[[INPUT]](%[[INPUT_SHAPE]] :
  // CHECK-SAME: !shapex.ranked_shape<[1,4,5,2]>) : f16,
  // CHECK-SAME: %[[FILTER]](%[[FILTER_SHAPE]] :
  // CHECK-SAME: !shapex.ranked_shape<[3,2,2,1]>) : f16,
  // CHECK-SAME: out %[[DST]](%[[DST_SHAPE]] :
  // CHECK-SAME: !shapex.ranked_shape<[1,2,3,1]>) : f16
  // CHECK-SAME: {batch_group_count = 1 : i32,
  // CHECK-SAME: feature_group_count = 1 : i32,
  // CHECK-SAME: lhs_dilation = dense<1> : vector<2xi32>,
  // CHECK-SAME: padding = dense<[1, 2, 2, 2]> : vector<4xi32>,
  // CHECK-SAME: rhs_dilation = dense<1> : vector<2xi32>,
  // CHECK-SAME: window_strides = dense<1> : vector<2xi32>}
 vmla.conv %input(%input_shape : !shapex.ranked_shape<[1,4,5,2]>) : f16,
           %filter(%filter_shape : !shapex.ranked_shape<[3,2,2,1]>) : f16,
           out %dst(%dst_shape : !shapex.ranked_shape<[1,2,3,1]>) : f16
           {batch_group_count = 1 : i32,
            feature_group_count = 1 : i32,
            lhs_dilation = dense<1> : vector<2xi32>,
            padding = dense<[1, 2, 2, 2]> : vector<4xi32>,
            rhs_dilation = dense<1> : vector<2xi32>,
            window_strides = dense<1> : vector<2xi32>}
  return
}

// CHECK-LABEL: @vmla_reduce
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[INIT:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[INIT_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_reduce(%src : !vmla.buffer,
                  %src_shape : !shapex.ranked_shape<[4,8]>,
                  %init : !vmla.buffer,
                  %init_shape : !shapex.ranked_shape<[]>,
                  %dst : !vmla.buffer,
                  %dst_shape : !shapex.ranked_shape<[4]>) {
  // CHECK-NEXT: vmla.reduce.sum
  // CEHCK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[4,8]>),
  // CHECK-SAME: %[[INIT]](%[[INIT_SHAPE]] : !shapex.ranked_shape<[]>),
  // CHECK-SAME: out %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[4]>)
  // CHECK-SaME: {dimension = 1 : i32} : f16
  vmla.reduce.sum %src(%src_shape : !shapex.ranked_shape<[4,8]>),
                  %init(%init_shape : !shapex.ranked_shape<[]>),
                  out %dst(%dst_shape : !shapex.ranked_shape<[4]>)
                  {dimension = 1 : i32} : f16
  return
}

// -----

// CHECK-LABEL: @vmla_pooling
// CHECK-SAME: %[[SRC:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SRC_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[INIT:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[INIT_SHAPE:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[DST_SHAPE:[a-zA-Z0-9$._-]+]]
func @vmla_pooling(%src : !vmla.buffer,
                   %src_shape : !shapex.ranked_shape<[32,32]>,
                   %init : !vmla.buffer,
                   %init_shape : !shapex.ranked_shape<[]>,
                   %dst : !vmla.buffer,
                   %dst_shape : !shapex.ranked_shape<[16,16]>) {
  // CHECK-NEXT: vmla.pooling.min
  // CEHCK-SAME: %[[SRC]](%[[SRC_SHAPE]] : !shapex.ranked_shape<[32,32]>),
  // CHECK-SAME: %[[INIT]](%[[INIT_SHAPE]] : !shapex.ranked_shape<[]>),
  // CHECK-SAME: out %[[DST]](%[[DST_SHAPE]] : !shapex.ranked_shape<[16,16]>)
  // CHECK-SAME: {padding = dense<0> : tensor<i32>,
  // CHECK-SaME:  window_dimensions = dense<[2,2]> : tensor<2xi32>,
  // CHECK-SaME:  window_strides = dense<[2,2]> : tensor<2xi32>} : f16
  vmla.pooling.min %src(%src_shape : !shapex.ranked_shape<[32,32]>),
                   %init(%init_shape : !shapex.ranked_shape<[]>),
                   out %dst(%dst_shape : !shapex.ranked_shape<[16,16]>)
                   {padding = dense<0> : tensor<i32>,
                    window_dimensions = dense<[2,2]> : tensor<2xi32>,
                    window_strides = dense<[2,2]> : tensor<2xi32>} : f16
  return
}
