// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-verify-vector-size-legality))" --split-input-file %s --verify-diagnostics

// expected-error @+1 {{One or more operations with large vector sizes (4096 bytes) were found:}}
func.func @large_vector_without_native_vector_size(%arg0 : vector<16x64xf32>) -> vector<16x64xf32> {
  // expected-note-re @+1 {{math.exp}}
  %0 = math.exp %arg0 : vector<16x64xf32>
  // expected-note-re @+1 {{return}}
  return %0 : vector<16x64xf32>
}

// -----

// expected-error @+1 {{One or more operations with large vector sizes (32768 bytes) were found:}}
func.func @large_vector_with_native_vector_size(%arg0 : vector<16x64x64xf32>) -> vector<16x64x64xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {native_vector_size = 64}>
} {
  // expected-note-re @+1 {{math.exp}}
  %0 = math.exp %arg0 : vector<16x64x64xf32>
  // expected-note-re @+1 {{return}}
  return %0 : vector<16x64x64xf32>
}

// -----

// expected-error @+1 {{One or more operations with large vector sizes (32768 bytes) were found:}}
func.func @large_contract_with_native_vector_size(%lhs : vector<32x64xf32>, %rhs : vector<32x64xf32>, %acc : vector<32x32xf32>) -> vector<32x32xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {native_vector_size = 64}>
} {
  // expected-note-re @+1 {{vector.contract}}
  %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<32x64xf32>, vector<32x64xf32> into vector<32x32xf32>
  return %0 : vector<32x32xf32>
}

// -----

// expected-error @+1 {{One or more operations with large vector sizes (4096 bytes) were found:}}
func.func @large_scalable_vector_without_native_vector_size(%arg0 : vector<[16]x64xf32>) -> vector<[16]x64xf32> {
  // expected-note-re @+1 {{math.exp}}
  %0 = math.exp %arg0 : vector<[16]x64xf32>
  // expected-note-re @+1 {{return}}
  return %0 : vector<[16]x64xf32>
}

// -----

// expected-error @+1 {{One or more operations with large vector sizes (8192 bytes) were found:}}
func.func @large_scalable_vector_with_native_vector_size(%arg0 : vector<[16]x64x64xf32>) -> vector<[16]x64x64xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {native_vector_size = 16}>
} {
  // expected-note-re @+1 {{math.exp}}
  %0 = math.exp %arg0 : vector<[16]x64x64xf32>
  // expected-note-re @+1 {{return}}
  return %0 : vector<[16]x64x64xf32>
}

// -----

// expected-error @+1 {{One or more operations with large vector sizes (8192 bytes) were found:}}
func.func @large_scalable_contract_with_native_vector_size(%lhs : vector<32x[64]xf32>, %rhs : vector<32x[64]xf32>, %acc : vector<32x32xf32>) -> vector<32x32xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {native_vector_size = 16}>
} {
  // expected-note-re @+1 {{vector.contract}}
  %0 = vector.contract {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>}
    %lhs, %rhs, %acc : vector<32x[64]xf32>, vector<32x[64]xf32> into vector<32x32xf32>
  return %0 : vector<32x32xf32>
}
