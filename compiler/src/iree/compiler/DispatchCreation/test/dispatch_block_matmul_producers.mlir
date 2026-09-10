// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions{aggressive=true}))" \
// RUN:   --iree-dispatch-creation-block-matmul-producer-fusion=true \
// RUN:   %s | FileCheck %s --check-prefix=CHECK-BLOCK
// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-clone-producers-into-dispatch-regions{aggressive=true}))" \
// RUN:   --iree-dispatch-creation-block-matmul-producer-fusion=false \
// RUN:   %s | FileCheck %s --check-prefix=CHECK-NO-BLOCK

// -----
// Base case: CPU target, ukernels are enabled,
#exec_target_cpu_mmt4d = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {ukernels = "all"}>
#device_target_cpu_mmt4d = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_cpu_mmt4d]> : !hal.device
util.global private @device_cpu_mmt4d = #device_target_cpu_mmt4d

util.func public @bitextend_feeds_matmul_blocked(%src : tensor<1x8xi8>, %rhs : tensor<8x1xf32>, %acc : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %cast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<1x8xi8>) outs(%empty : tensor<1x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %ext = arith.sitofp %in : i8 to f32
    linalg.yield %ext : f32
  } -> tensor<1x8xf32>
  %0 = flow.dispatch.region -> (tensor<1x1xf32>) {
    %matmul = linalg.matmul ins(%cast, %rhs : tensor<1x8xf32>, tensor<8x1xf32>) outs(%acc : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.return %matmul : tensor<1x1xf32>
  }
  util.return %0 : tensor<1x1xf32>
}
// Blocking forces two dispatch regions
// CHECK-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_blocked
//       CHECK-BLOCK:   %[[GENDISP:.+]] = flow.dispatch.region
//       CHECK-BLOCK:     linalg.generic
//       CHECK-BLOCK:     flow.return
//       CHECK-BLOCK:   %[[MATMULDISP:.+]] = flow.dispatch.region
//       CHECK-BLOCK:     linalg.matmul
//  CHECK-BLOCK-SAME:       ins(%[[GENDISP]]
//       CHECK-BLOCK:     flow.return
//       CHECK-BLOCK:   util.return %[[MATMULDISP]]

// Without blocking both ops are fused into the same region
// CHECK-NO-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_blocked
//       CHECK-NO-BLOCK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-NO-BLOCK:     linalg.matmul
//  CHECK-NO-BLOCK-SAME:       ins(%[[CAST]]
//       CHECK-NO-BLOCK:   util.return %[[DISPATCH]]

// -----
// No ukernels, no blocking
#exec_target_no_ukernel = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {ukernels = "none"}>
#device_target_no_ukernel = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_no_ukernel]> : !hal.device
util.global private @device_no_ukernel = #device_target_no_ukernel

util.func public @bitextend_feeds_matmul_no_ukernel_not_blocked(%src : tensor<1x8xi8>, %rhs : tensor<8x1xf32>, %acc : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %cast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<1x8xi8>) outs(%empty : tensor<1x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %ext = arith.sitofp %in : i8 to f32
    linalg.yield %ext : f32
  } -> tensor<1x8xf32>
  %0 = flow.dispatch.region -> (tensor<1x1xf32>) {
    %matmul = linalg.matmul ins(%cast, %rhs : tensor<1x8xf32>, tensor<8x1xf32>) outs(%acc : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.return %matmul : tensor<1x1xf32>
  }
  util.return %0 : tensor<1x1xf32>
}
// CHECK-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_no_ukernel_not_blocked
//       CHECK-BLOCK:   flow.dispatch.region
//       CHECK-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-BLOCK:     linalg.matmul
//  CHECK-BLOCK-SAME:       ins(%[[CAST]]
// CHECK-NO-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_no_ukernel_not_blocked
//       CHECK-NO-BLOCK:   flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-NO-BLOCK:     linalg.matmul
//  CHECK-NO-BLOCK-SAME:       ins(%[[CAST]]

// -----
// No CPU, no blocking
#exec_target_vmvx = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
#device_target_vmvx = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_vmvx]> : !hal.device
util.global private @device_vmvx = #device_target_vmvx

util.func public @bitextend_feeds_matmul_non_llvmcpu_not_blocked(%src : tensor<1x8xi8>, %rhs : tensor<8x1xf32>, %acc : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %cast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<1x8xi8>) outs(%empty : tensor<1x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %ext = arith.sitofp %in : i8 to f32
    linalg.yield %ext : f32
  } -> tensor<1x8xf32>
  %0 = flow.dispatch.region -> (tensor<1x1xf32>) {
    %matmul = linalg.matmul ins(%cast, %rhs : tensor<1x8xf32>, tensor<8x1xf32>) outs(%acc : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.return %matmul : tensor<1x1xf32>
  }
  util.return %0 : tensor<1x1xf32>
}
// CHECK-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_non_llvmcpu_not_blocked
//       CHECK-BLOCK:   flow.dispatch.region
//       CHECK-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-BLOCK:     linalg.matmul
//  CHECK-BLOCK-SAME:       ins(%[[CAST]]
// CHECK-NO-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_non_llvmcpu_not_blocked
//       CHECK-NO-BLOCK:   flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-NO-BLOCK:     linalg.matmul
//  CHECK-NO-BLOCK-SAME:       ins(%[[CAST]]

// -----
// Multiple devices, no blocking
#exec_target_a = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {ukernels = "all"}>
#exec_target_b = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {ukernels = "all"}>
#device_target_a = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_a]> : !hal.device
#device_target_b = #hal.device.target<"local", {ordinal = 1 : index}, [#exec_target_b]> : !hal.device
util.global private @device_a = #device_target_a
util.global private @device_b = #device_target_b

util.func public @bitextend_feeds_matmul_heterogeneous_devices_not_blocked(%src : tensor<1x8xi8>, %rhs : tensor<8x1xf32>, %acc : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %cast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<1x8xi8>) outs(%empty : tensor<1x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %ext = arith.sitofp %in : i8 to f32
    linalg.yield %ext : f32
  } -> tensor<1x8xf32>
  %0 = flow.dispatch.region -> (tensor<1x1xf32>) {
    %matmul = linalg.matmul ins(%cast, %rhs : tensor<1x8xf32>, tensor<8x1xf32>) outs(%acc : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.return %matmul : tensor<1x1xf32>
  }
  util.return %0 : tensor<1x1xf32>
}
// CHECK-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_heterogeneous_devices_not_blocked
//       CHECK-BLOCK:   flow.dispatch.region
//       CHECK-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-BLOCK:     linalg.matmul
//  CHECK-BLOCK-SAME:       ins(%[[CAST]]
// CHECK-NO-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_heterogeneous_devices_not_blocked
//       CHECK-NO-BLOCK:   flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-NO-BLOCK:     linalg.matmul
//  CHECK-NO-BLOCK-SAME:       ins(%[[CAST]]

// -----
// No contraction consumer, no blocking
#exec_target_cpu_mmt4d = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {ukernels = "all"}>
#device_target_cpu_mmt4d = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_cpu_mmt4d]> : !hal.device
util.global private @device_cpu_mmt4d = #device_target_cpu_mmt4d

util.func public @bitextend_feeds_non_contraction_consumer_not_blocked(%src : tensor<1x8xi8>) -> tensor<1x8xf32> {
  %empty = tensor.empty() : tensor<1x8xf32>
  %cast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<1x8xi8>) outs(%empty : tensor<1x8xf32>) {
  ^bb0(%in: i8, %out: f32):
    %ext = arith.sitofp %in : i8 to f32
    linalg.yield %ext : f32
  } -> tensor<1x8xf32>
  %0 = flow.dispatch.region -> (tensor<1x8xf32>) {
    %doubled = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%cast : tensor<1x8xf32>) outs(%empty : tensor<1x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %doubled_elem = arith.addf %in, %in : f32
      linalg.yield %doubled_elem : f32
    } -> tensor<1x8xf32>
    flow.return %doubled : tensor<1x8xf32>
  }
  util.return %0 : tensor<1x8xf32>
}
// CHECK-BLOCK-LABEL: util.func public @bitextend_feeds_non_contraction_consumer_not_blocked
//       CHECK-BLOCK:   flow.dispatch.region
//       CHECK-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-BLOCK:     linalg.generic
//  CHECK-BLOCK-SAME:       ins(%[[CAST]]
// CHECK-NO-BLOCK-LABEL: util.func public @bitextend_feeds_non_contraction_consumer_not_blocked
//       CHECK-NO-BLOCK:   flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[CAST:.+]] = linalg.generic
//       CHECK-NO-BLOCK:     linalg.generic
//  CHECK-NO-BLOCK-SAME:       ins(%[[CAST]]

// -----
// matmul consumer through reshape
#exec_target_cpu_mmt4d = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {ukernels = "all"}>
#device_target_cpu_mmt4d = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_cpu_mmt4d]> : !hal.device
util.global private @device_cpu_mmt4d = #device_target_cpu_mmt4d

util.func public @bitextend_feeds_matmul_through_reshape_chain_blocked(%src : tensor<2x4xi8>, %rhs : tensor<8x1xf32>, %acc : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<2x4xf32>
  %cast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%src : tensor<2x4xi8>) outs(%empty : tensor<2x4xf32>) {
  ^bb0(%in: i8, %out: f32):
    %ext = arith.sitofp %in : i8 to f32
    linalg.yield %ext : f32
  } -> tensor<2x4xf32>
  %0 = flow.dispatch.region -> (tensor<1x1xf32>) {
    %collapsed = tensor.collapse_shape %cast [[0, 1]] : tensor<2x4xf32> into tensor<8xf32>
    %expanded = tensor.expand_shape %collapsed [[0, 1]] output_shape [1, 8] : tensor<8xf32> into tensor<1x8xf32>
    %matmul = linalg.matmul ins(%expanded, %rhs : tensor<1x8xf32>, tensor<8x1xf32>) outs(%acc : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.return %matmul : tensor<1x1xf32>
  }
  util.return %0 : tensor<1x1xf32>
}
// CHECK-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_through_reshape_chain_blocked
//       CHECK-BLOCK:   %[[GENDISP:.+]] = flow.dispatch.region
//       CHECK-BLOCK:     linalg.generic
//       CHECK-BLOCK:     flow.return
//       CHECK-BLOCK:   %[[MATMULDISP:.+]] = flow.dispatch.region
//       CHECK-BLOCK:     tensor.collapse_shape
//  CHECK-BLOCK-SAME:       %[[GENDISP]]
//       CHECK-BLOCK:     linalg.matmul
//       CHECK-BLOCK:     flow.return
//       CHECK-BLOCK:   util.return %[[MATMULDISP]]

// CHECK-NO-BLOCK-LABEL: util.func public @bitextend_feeds_matmul_through_reshape_chain_blocked
//       CHECK-NO-BLOCK:   %[[MATMULDISP:.+]] = flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[GEN:.+]] = linalg.generic
//       CHECK-NO-BLOCK:     tensor.collapse_shape
//  CHECK-NO-BLOCK-SAME:       %[[GEN]]
//       CHECK-NO-BLOCK:     linalg.matmul
//       CHECK-NO-BLOCK:     flow.return
//       CHECK-NO-BLOCK:   util.return %[[MATMULDISP]]

// -----
// gather feeds into matmul
#exec_target_cpu_mmt4d = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {ukernels = "all"}>
#device_target_cpu_mmt4d = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_cpu_mmt4d]> : !hal.device
util.global private @device_cpu_mmt4d = #device_target_cpu_mmt4d

util.func public @gather_feeds_matmul_blocked(%source : tensor<2x2x8xf32>, %indices : tensor<2xi32>, %rhs : tensor<8x1xf32>, %acc : tensor<1x1xf32>) -> tensor<1x1xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %gathered = iree_linalg_ext.gather dimension_map = [0, 1]
                          ins(%source, %indices : tensor<2x2x8xf32>, tensor<2xi32>)
                          outs(%empty: tensor<8xf32>) -> tensor<8xf32>
  %0 = flow.dispatch.region -> (tensor<1x1xf32>) {
    %expanded = tensor.expand_shape %gathered [[0, 1]] output_shape [1, 8] : tensor<8xf32> into tensor<1x8xf32>
    %matmul = linalg.matmul ins(%expanded, %rhs : tensor<1x8xf32>, tensor<8x1xf32>) outs(%acc : tensor<1x1xf32>) -> tensor<1x1xf32>
    flow.return %matmul : tensor<1x1xf32>
  }
  util.return %0 : tensor<1x1xf32>
}
// CHECK-BLOCK-LABEL: util.func public @gather_feeds_matmul_blocked
//       CHECK-BLOCK:   %[[GATHERDISP:.+]] = flow.dispatch.region
//       CHECK-BLOCK:     iree_linalg_ext.gather
//       CHECK-BLOCK:     flow.return
//       CHECK-BLOCK:   %[[MATMULDISP:.+]] = flow.dispatch.region
//       CHECK-BLOCK:     tensor.expand_shape
//  CHECK-BLOCK-SAME:       %[[GATHERDISP]]
//       CHECK-BLOCK:     linalg.matmul
//       CHECK-BLOCK:     flow.return
//       CHECK-BLOCK:   util.return %[[MATMULDISP]]

// CHECK-NO-BLOCK-LABEL: util.func public @gather_feeds_matmul_blocked
//       CHECK-NO-BLOCK:   %[[MATMULDISP:.+]] = flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[GATHER:.+]] = iree_linalg_ext.gather
//       CHECK-NO-BLOCK:     tensor.expand_shape
//  CHECK-NO-BLOCK-SAME:       %[[GATHER]]
//       CHECK-NO-BLOCK:     linalg.matmul
//       CHECK-NO-BLOCK:     flow.return
//       CHECK-NO-BLOCK:   util.return %[[MATMULDISP]]

// -----
// gather feeds to non-contraction
#exec_target_cpu_mmt4d = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {ukernels = "all"}>
#device_target_cpu_mmt4d = #hal.device.target<"local", {ordinal = 0 : index}, [#exec_target_cpu_mmt4d]> : !hal.device
util.global private @device_cpu_mmt4d = #device_target_cpu_mmt4d

util.func public @gather_feeds_non_contraction_consumer_not_blocked(%source : tensor<2x2x8xf32>, %indices : tensor<2xi32>) -> tensor<8xf32> {
  %empty = tensor.empty() : tensor<8xf32>
  %gathered = iree_linalg_ext.gather dimension_map = [0, 1]
                          ins(%source, %indices : tensor<2x2x8xf32>, tensor<2xi32>)
                          outs(%empty: tensor<8xf32>) -> tensor<8xf32>
  %0 = flow.dispatch.region -> (tensor<8xf32>) {
    %doubled = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        ins(%gathered : tensor<8xf32>) outs(%empty : tensor<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %doubled_elem = arith.addf %in, %in : f32
      linalg.yield %doubled_elem : f32
    } -> tensor<8xf32>
    flow.return %doubled : tensor<8xf32>
  }
  util.return %0 : tensor<8xf32>
}
// CHECK-BLOCK-LABEL: util.func public @gather_feeds_non_contraction_consumer_not_blocked
//       CHECK-BLOCK:   flow.dispatch.region
//       CHECK-BLOCK:     %[[GATHERED:.+]] = iree_linalg_ext.gather
//       CHECK-BLOCK:     linalg.generic
//  CHECK-BLOCK-SAME:       ins(%[[GATHERED]]

// CHECK-NO-BLOCK-LABEL: util.func public @gather_feeds_non_contraction_consumer_not_blocked
//       CHECK-NO-BLOCK:   flow.dispatch.region
//       CHECK-NO-BLOCK:     %[[GATHERED:.+]] = iree_linalg_ext.gather
//       CHECK-NO-BLOCK:     linalg.generic
//  CHECK-NO-BLOCK-SAME:       ins(%[[GATHERED]]
