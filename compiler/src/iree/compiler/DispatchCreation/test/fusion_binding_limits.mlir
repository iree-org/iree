// RUN: iree-opt --split-input-file --iree-dispatch-creation-elementwise-op-fusion %s | FileCheck %s --check-prefixes=CHECK,FUSION
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-fuse-multi-use-elementwise-producer))' %s | FileCheck %s --check-prefixes=CHECK,FUSION
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))' %s | FileCheck %s --check-prefixes=CHECK,REGIONS

// Both inputs and outputs contribute to the target binding budget.
#id = affine_map<(d0) -> (d0)>
module attributes {hal.device.targets = [#hal.device.target<"test", [#hal.executable.target<"test", "test-format", {max_dispatch_binding_count = 8 : i64}>]> : !hal.device]} {
// CHECK-LABEL: util.func public @binding_limit_eight
util.func public @binding_limit_eight(%a0: tensor<32xf32>, %a1: tensor<32xf32>, %a2: tensor<32xf32>, %a3: tensor<32xf32>, %a4: tensor<32xf32>, %a5: tensor<32xf32>, %a6: tensor<32xf32>, %a7: tensor<32xf32>) -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%a0, %a1, %a2, %a3 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    linalg.yield %v3 : f32
  } -> tensor<32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%sum, %a4, %a5, %a6, %a7 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    %v4 = arith.addf %v3, %x4 : f32
    linalg.yield %v4 : f32
  } -> tensor<32xf32>
  util.return %result : tensor<32xf32>
}
// FUSION-COUNT-2: linalg.generic
// FUSION-NOT: linalg.generic
// REGIONS-COUNT-2: flow.dispatch.region
// REGIONS-NOT: flow.dispatch.region
// CHECK-LABEL: util.func public @binding_limit_exactly_eight
util.func public @binding_limit_exactly_eight(%a0: tensor<32xf32>, %a1: tensor<32xf32>, %a2: tensor<32xf32>, %a3: tensor<32xf32>, %a4: tensor<32xf32>, %a5: tensor<32xf32>, %a6: tensor<32xf32>) -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%a0, %a1, %a2, %a3 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    linalg.yield %v3 : f32
  } -> tensor<32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%sum, %a4, %a5, %a6 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    linalg.yield %v3 : f32
  } -> tensor<32xf32>
  util.return %result : tensor<32xf32>
}
// FUSION-COUNT-1: linalg.generic
// FUSION-NOT: linalg.generic
// REGIONS-COUNT-1: flow.dispatch.region
// REGIONS-NOT: flow.dispatch.region
// CHECK-LABEL: util.func public @binding_limit_below_eight
util.func public @binding_limit_below_eight(%a0: tensor<32xf32>, %a1: tensor<32xf32>, %a2: tensor<32xf32>, %a3: tensor<32xf32>, %a4: tensor<32xf32>) -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%a0, %a1, %a2, %a3 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    linalg.yield %v3 : f32
  } -> tensor<32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]}
      ins(%sum, %a4 : tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    linalg.yield %v1 : f32
  } -> tensor<32xf32>
  util.return %result : tensor<32xf32>
}
// FUSION-COUNT-1: linalg.generic
// FUSION-NOT: linalg.generic
// REGIONS-COUNT-1: flow.dispatch.region
// REGIONS-NOT: flow.dispatch.region
}

// -----

// Missing configuration retains the existing fusion limit.
#id = affine_map<(d0) -> (d0)>
// CHECK-LABEL: util.func public @binding_limit_unspecified
util.func public @binding_limit_unspecified(%a0: tensor<32xf32>, %a1: tensor<32xf32>, %a2: tensor<32xf32>, %a3: tensor<32xf32>, %a4: tensor<32xf32>, %a5: tensor<32xf32>, %a6: tensor<32xf32>, %a7: tensor<32xf32>) -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%a0, %a1, %a2, %a3 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    linalg.yield %v3 : f32
  } -> tensor<32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%sum, %a4, %a5, %a6, %a7 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    %v4 = arith.addf %v3, %x4 : f32
    linalg.yield %v4 : f32
  } -> tensor<32xf32>
  util.return %result : tensor<32xf32>
}
// FUSION-COUNT-1: linalg.generic
// FUSION-NOT: linalg.generic
// REGIONS-COUNT-1: flow.dispatch.region
// REGIONS-NOT: flow.dispatch.region

// -----

// Mixed targets use the smallest limit; explicit affinities select a device.
#id = affine_map<(d0) -> (d0)>
module attributes {hal.device.targets = [#hal.device.target<"test", [#hal.executable.target<"test", "test-format", {max_dispatch_binding_count = 16 : i64}>, #hal.executable.target<"test", "test-format", {max_dispatch_binding_count = 8 : i64}>]> : !hal.device]} {
  util.global private @cpu = #hal.device.target<"test", [#hal.executable.target<"test", "test-format", {max_dispatch_binding_count = 16 : i64}>]> : !hal.device
// CHECK-LABEL: util.func public @binding_limit_mixed_targets
util.func public @binding_limit_mixed_targets(%a0: tensor<32xf32>, %a1: tensor<32xf32>, %a2: tensor<32xf32>, %a3: tensor<32xf32>, %a4: tensor<32xf32>, %a5: tensor<32xf32>, %a6: tensor<32xf32>, %a7: tensor<32xf32>) -> tensor<32xf32> {
  %empty = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%a0, %a1, %a2, %a3 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    linalg.yield %v3 : f32
  } -> tensor<32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%sum, %a4, %a5, %a6, %a7 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    %v4 = arith.addf %v3, %x4 : f32
    linalg.yield %v4 : f32
  } -> tensor<32xf32>
  util.return %result : tensor<32xf32>
}
// FUSION-COUNT-2: linalg.generic
// FUSION-NOT: linalg.generic
// REGIONS-COUNT-2: flow.dispatch.region
// REGIONS-NOT: flow.dispatch.region
// CHECK-LABEL: util.func public @binding_limit_device_affinity
util.func public @binding_limit_device_affinity(%a0: tensor<32xf32>, %a1: tensor<32xf32>, %a2: tensor<32xf32>, %a3: tensor<32xf32>, %a4: tensor<32xf32>, %a5: tensor<32xf32>, %a6: tensor<32xf32>, %a7: tensor<32xf32>) -> tensor<32xf32> attributes {stream.affinity.default = #hal.device.affinity<@cpu>} {
  %empty = tensor.empty() : tensor<32xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%a0, %a1, %a2, %a3 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    linalg.yield %v3 : f32
  } -> tensor<32xf32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id, #id, #id, #id], iterator_types = ["parallel"]}
      ins(%sum, %a4, %a5, %a6, %a7 : tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) outs(%empty : tensor<32xf32>) {
  ^bb0(%x0: f32, %x1: f32, %x2: f32, %x3: f32, %x4: f32, %unused: f32):
    %v1 = arith.addf %x0, %x1 : f32
    %v2 = arith.addf %v1, %x2 : f32
    %v3 = arith.addf %v2, %x3 : f32
    %v4 = arith.addf %v3, %x4 : f32
    linalg.yield %v4 : f32
  } -> tensor<32xf32>
  util.return %result : tensor<32xf32>
}
// FUSION-COUNT-1: linalg.generic
// FUSION-NOT: linalg.generic
// REGIONS-COUNT-1: flow.dispatch.region
// REGIONS-NOT: flow.dispatch.region
}
