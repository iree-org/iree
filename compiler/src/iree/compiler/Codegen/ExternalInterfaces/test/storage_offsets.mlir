// RUN: iree-opt --split-input-file --iree-stream-encode-host-tensors %s | FileCheck %s

// -----

#layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [2, 4], outerDimsPerm = [1, 0]}}>
// CHECK-LABEL: @cpu_tiled_load
// CHECK: %[[OFFSET:.*]] = arith.constant 116 : index
// CHECK: stream.async.load {{.*}}[%[[OFFSET]]]
util.func public @cpu_tiled_load(%buffer: !stream.resource<staging>) -> i32 {
  %size = arith.constant 192 : index
  %row = arith.constant 3 : index
  %col = arith.constant 5 : index
  %value = stream.tensor.load %buffer[%row, %col] : tensor<4x12xi32, #layout> in !stream.resource<staging>{%size} -> i32
  util.return %value : i32
}

// -----

#layout = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [2, 4], outerDimsPerm = [1, 0]}}>
// CHECK-LABEL: @gpu_tiled_load
// CHECK: %[[OFFSET:.*]] = arith.constant 116 : index
// CHECK: stream.async.load {{.*}}[%[[OFFSET]]]
util.func public @gpu_tiled_load(%buffer: !stream.resource<staging>) -> i32 {
  %size = arith.constant 192 : index
  %row = arith.constant 3 : index
  %col = arith.constant 5 : index
  %value = stream.tensor.load %buffer[%row, %col] : tensor<4x12xi32, #layout> in !stream.resource<staging>{%size} -> i32
  util.return %value : i32
}

// -----

#layout = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [2, 4], outerDimsPerm = [1, 0]}}>
// CHECK-LABEL: @vmvx_tiled_load
// CHECK: %[[OFFSET:.*]] = arith.constant 116 : index
// CHECK: stream.async.load {{.*}}[%[[OFFSET]]]
util.func public @vmvx_tiled_load(%buffer: !stream.resource<staging>) -> i32 {
  %size = arith.constant 192 : index
  %row = arith.constant 3 : index
  %col = arith.constant 5 : index
  %value = stream.tensor.load %buffer[%row, %col] : tensor<4x12xi32, #layout> in !stream.resource<staging>{%size} -> i32
  util.return %value : i32
}

// -----

// Inner index [1, 1] expands to [1, 0, 1] and permutes to [1, 1, 0], giving
// offset 6 within the tile. The byte offset is (24 + 6) * 4 = 120.
#layout = #iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [2, 4], outerDimsPerm = [1, 0], swizzle = {expandShape = [[["Internal", 2]], [["Internal", 2], ["Internal", 2]]], permutation = [0, 2, 1]}}}>
// CHECK-LABEL: @swizzled_load
// CHECK: %[[OFFSET:.*]] = arith.constant 120 : index
// CHECK: %[[VALUE:.*]] = stream.async.load %arg0[%[[OFFSET]]] {{.*}} -> i32
// CHECK: util.return %[[VALUE]] : i32
util.func public @swizzled_load(%buffer: !stream.resource<staging>) -> i32 {
  %size = arith.constant 192 : index
  %row = arith.constant 3 : index
  %col = arith.constant 5 : index
  %value = stream.tensor.load %buffer[%row, %col] : tensor<4x12xi32, #layout> in !stream.resource<staging>{%size} -> i32
  util.return %value : i32
}

// CHECK-LABEL: @swizzled_store
// CHECK: %[[OFFSET:.*]] = arith.constant 120 : index
// CHECK: stream.async.store %arg1, {{.*}}[%[[OFFSET]]]
util.func public @swizzled_store(%buffer: !stream.resource<staging>, %value: i32) -> !stream.resource<staging> {
  %size = arith.constant 192 : index
  %row = arith.constant 3 : index
  %col = arith.constant 5 : index
  %result = stream.tensor.store %value, %buffer[%row, %col] : i32 -> tensor<4x12xi32, #layout> in %buffer as !stream.resource<staging>{%size}
  util.return %result : !stream.resource<staging>
}

// -----

#layout = #iree_cpu.vmvx_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0], innerTileSizes = [-1], outerDimsPerm = [0]}}>
// CHECK-LABEL: @do_not_guess_dynamic_tile
// CHECK-NOT: stream.async.load
// CHECK: stream.tensor.load
util.func public @do_not_guess_dynamic_tile(%buffer: !stream.resource<staging>, %index: index) -> i32 {
  %size = arith.constant 128 : index
  %value = stream.tensor.load %buffer[%index] : tensor<32xi32, #layout> in !stream.resource<staging>{%size} -> i32
  util.return %value : i32
}

// -----

#layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0], innerTileSizes = [4], scalableTiles = [true], outerDimsPerm = [0]}}>
// CHECK-LABEL: @do_not_guess_scalable_tile
// CHECK-NOT: stream.async.load
// CHECK: stream.tensor.load
util.func public @do_not_guess_scalable_tile(%buffer: !stream.resource<staging>, %index: index) -> i32 {
  %size = arith.constant 128 : index
  %value = stream.tensor.load %buffer[%index] : tensor<32xi32, #layout> in !stream.resource<staging>{%size} -> i32
  util.return %value : i32
}

// -----

// Five rows require three tiles of two rows each.
#layout = #iree_cpu.cpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [2, 4], outerDimsPerm = [1, 0]}}>
// CHECK-LABEL: @dynamic_tiled_load
// CHECK: %[[OFFSET:.*]] = arith.constant 148 : index
// CHECK: stream.async.load {{.*}}[%[[OFFSET]]]
util.func public @dynamic_tiled_load(%buffer: !stream.resource<staging>) -> i32 {
  %size = arith.constant 288 : index
  %row = arith.constant 3 : index
  %col = arith.constant 5 : index
  %rows = arith.constant 5 : index
  %value = stream.tensor.load %buffer[%row, %col] : tensor<?x12xi32, #layout>{%rows} in !stream.resource<staging>{%size} -> i32
  util.return %value : i32
}
