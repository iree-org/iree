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
