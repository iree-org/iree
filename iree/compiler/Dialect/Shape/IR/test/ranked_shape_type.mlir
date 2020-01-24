// RUN: iree-opt -split-input-file -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: @parseScalarShape
// CHECK: !shape.ranked_shape<i32>
func @parseScalarShape(%arg0 : !shape.ranked_shape<i32>) {
  return
}

// -----
// CHECK-LABEL: @parseStaticShape
// CHECK: !shape.ranked_shape<1x2xi32>
func @parseStaticShape(%arg0 : !shape.ranked_shape<1x2xi32>) {
  return
}

// -----
// CHECK-LABEL: @parseDynamicShape
// CHECK: !shape.ranked_shape<1x?x2x?xi32>
func @parseDynamicShape(%arg0 : !shape.ranked_shape<1x?x2x?xi32>) {
  return
}

// -----
// expected-error @+1 {{RankedShapeType must have an integral dim type}}
func @error(%arg0 : !shape.ranked_shape<1x?xf32>) {
  return
}
