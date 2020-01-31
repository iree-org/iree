// RUN: iree-opt -split-input-file -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: @parseScalarShapeIndex
// CHECK: !shape.ranked_shape<[]>
func @parseScalarShapeIndex(%arg0 : !shape.ranked_shape<[]>) {
  return
}

// -----
// CHECK-LABEL: @parseStaticShapeIndex
// CHECK: !shape.ranked_shape<[1,2]>
func @parseStaticShapeIndex(%arg0 : !shape.ranked_shape<[1, 2]>) {
  return
}

// CHECK-LABEL: @parseScalarShape
// CHECK: !shape.ranked_shape<[],i32>
func @parseScalarShape(%arg0 : !shape.ranked_shape<[], i32>) {
  return
}

// -----
// CHECK-LABEL: @parseStaticShape
// CHECK: !shape.ranked_shape<[1,2],i32>
func @parseStaticShape(%arg0 : !shape.ranked_shape<[1, 2], i32>) {
  return
}

// -----
// CHECK-LABEL: @parseDynamicShape
// CHECK: !shape.ranked_shape<[1,?,2,?],i32>
func @parseDynamicShape(%arg0 : !shape.ranked_shape<[1,?,2,?],i32>) {
  return
}

// -----
// expected-error @+1 {{RankedShapeType must have an integral or index dim type}}
func @error(%arg0 : !shape.ranked_shape<[1,?],f32>) {
  return
}
