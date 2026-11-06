// RUN: iree-compile --iree-input-type=stablehlo \
// RUN:   --compile-to=executable-configurations \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-cpu=generic %s \
// RUN:   | FileCheck %s --implicit-check-not=64x1024 \
// RUN:   --implicit-check-not=linalg.softmax

// SDPA reaches online attention on CPU, which walks the 1024 keys in tiles, so
// no 64x1024 score tensor exists. The decomposition is placeholder plumbing.
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: lowering_config = #iree_cpu.lowering_config<{{.*}}vector_reduction = [0, 0, 0, {{[0-9]+}}, {{[1-9][0-9]?[0-9]?}}]>
func.func @sdpa_long_keys(%q: tensor<2x3x64x32xf32>, %k: tensor<2x3x1024x32xf32>, %v: tensor<2x3x1024x48xf32>) -> tensor<2x3x64x48xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = false},
    decomposition = @sdpa_impl, version = 1 : i32
  } : (tensor<2x3x64x32xf32>, tensor<2x3x1024x32xf32>, tensor<2x3x1024x48xf32>) -> tensor<2x3x64x48xf32>
  return %0 : tensor<2x3x64x48xf32>
}
func.func private @sdpa_impl(%q: tensor<2x3x64x32xf32>, %k: tensor<2x3x1024x32xf32>, %v: tensor<2x3x1024x48xf32>) -> tensor<2x3x64x48xf32> {
  %s = stablehlo.dot_general %q, %k, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [3] : (tensor<2x3x64x32xf32>, tensor<2x3x1024x32xf32>) -> tensor<2x3x64x1024xf32>
  %o = stablehlo.dot_general %s, %v, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<2x3x64x1024xf32>, tensor<2x3x1024x48xf32>) -> tensor<2x3x64x48xf32>
  return %o : tensor<2x3x64x48xf32>
}

// A causal SDPA masks in the region, so no 64x1024 mask tensor exists either.
// The two batch dims collapse into one, so query and key are indices 1 and 2.
// CHECK: @sdpa_causal_long_keys_dispatch
// CHECK: iree_linalg_ext.online_attention
// CHECK-SAME: lowering_config = #iree_cpu.lowering_config<{{.*}}vector_reduction = [0, 0, 0, {{[0-9]+}}, {{[1-9][0-9]?[0-9]?}}]>
// CHECK: %[[QUERY:.+]] = iree_linalg_ext.index 1
// CHECK: %[[KEY:.+]] = iree_linalg_ext.index 2
// CHECK: arith.cmpi ugt, %[[KEY]], %[[QUERY]]
func.func @sdpa_causal_long_keys(%q: tensor<2x3x64x32xf32>, %k: tensor<2x3x1024x32xf32>, %v: tensor<2x3x1024x48xf32>) -> tensor<2x3x64x48xf32> {
  %0 = stablehlo.composite "torch.scaled_dot_product_attention" %q, %k, %v {
    composite_attributes = {enable_gqa = false, is_causal = true},
    decomposition = @sdpa_impl, version = 1 : i32
  } : (tensor<2x3x64x32xf32>, tensor<2x3x1024x32xf32>, tensor<2x3x1024x48xf32>) -> tensor<2x3x64x48xf32>
  return %0 : tensor<2x3x64x48xf32>
}
