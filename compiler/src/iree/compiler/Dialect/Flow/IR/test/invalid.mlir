// RUN: iree-opt --split-input-file --verify-diagnostics %s

// Test: encoding expects 1 encoding dim but 0 provided (encode direction).
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, 128, 64]>
util.func public @encode_missing_encoding_dims(%arg0 : tensor<?x64xf32>, %m : index) -> tensor<?x64xf32, #encoding> {
  // expected-error @+1 {{encoding expects 1 encoding dim(s), but 0 provided}}
  %0 = flow.tensor.encode %arg0 : tensor<?x64xf32>{%m} -> tensor<?x64xf32, #encoding>{%m}
  util.return %0 : tensor<?x64xf32, #encoding>
}

// -----

// Test: encoding expects 1 encoding dim but 2 provided (encode direction).
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, 128, 64]>
util.func public @encode_too_many_encoding_dims(%arg0 : tensor<?x64xf32>, %m : index) -> tensor<?x64xf32, #encoding> {
  // expected-error @+1 {{encoding expects 1 encoding dim(s), but 2 provided}}
  %0 = flow.tensor.encode %arg0 encoding_dims{%m, %m} : tensor<?x64xf32>{%m} -> tensor<?x64xf32, #encoding>{%m}
  util.return %0 : tensor<?x64xf32, #encoding>
}

// -----

// Test: encoding expects 1 encoding dim but 0 provided (decode direction).
#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2], iteration_sizes = [?, 128, 64]>
util.func public @decode_missing_encoding_dims(%arg0 : tensor<?x64xf32, #encoding>, %m : index) -> tensor<?x64xf32> {
  // expected-error @+1 {{encoding expects 1 encoding dim(s), but 0 provided}}
  %0 = flow.tensor.encode %arg0 : tensor<?x64xf32, #encoding>{%m} -> tensor<?x64xf32>{%m}
  util.return %0 : tensor<?x64xf32>
}
