// RUN: iree-opt -split-input-file -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: @parseScalarShape
// CHECK: !iree.ranked_shape<i32>
func @parseScalarShape(%arg0 : !iree.ranked_shape<i32>) {
  return
}

// -----
// CHECK-LABEL: @parseStaticShape
// CHECK: !iree.ranked_shape<1x2xi32>
func @parseStaticShape(%arg0 : !iree.ranked_shape<1x2xi32>) {
  return
}

// -----
// CHECK-LABEL: @parseDynamicShape
// CHECK: !iree.ranked_shape<1x?x2x?xi32>
func @parseDynamicShape(%arg0 : !iree.ranked_shape<1x?x2x?xi32>) {
  return
}

// -----
// expected-error @+1 {{RankedShapeType must have an integral dim type}}
func @error(%arg0 : !iree.ranked_shape<1x?xf32>) {
  return
}
