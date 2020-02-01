// RUN: iree-opt -split-input-file -verify-diagnostics %s | IreeFileCheck %s

// CHECK-LABEL: @parseScalarShapeIndex
// CHECK: !shapex.ranked_shape<[]>
func @parseScalarShapeIndex(%arg0 : !shapex.ranked_shape<[]>) {
  return
}

// -----
// CHECK-LABEL: @parseStaticShapeIndex
// CHECK: !shapex.ranked_shape<[1,2]>
func @parseStaticShapeIndex(%arg0 : !shapex.ranked_shape<[1, 2]>) {
  return
}

// CHECK-LABEL: @parseScalarShape
// CHECK: !shapex.ranked_shape<[],i32>
func @parseScalarShape(%arg0 : !shapex.ranked_shape<[], i32>) {
  return
}

// -----
// CHECK-LABEL: @parseStaticShape
// CHECK: !shapex.ranked_shape<[1,2],i32>
func @parseStaticShape(%arg0 : !shapex.ranked_shape<[1, 2], i32>) {
  return
}

// -----
// CHECK-LABEL: @parseDynamicShape
// CHECK: !shapex.ranked_shape<[1,?,2,?],i32>
func @parseDynamicShape(%arg0 : !shapex.ranked_shape<[1,?,2,?],i32>) {
  return
}

// -----
// expected-error @+1 {{RankedShapeType must have an integral or index dim type}}
func @error(%arg0 : !shapex.ranked_shape<[1,?],f32>) {
  return
}
