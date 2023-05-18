// RUN: iree-opt --split-input-file --canonicalize=test-convergence=true %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldChannelRankOp
//  CHECK-SAME: (%[[RANK:.+]]: index)
func.func @FoldChannelRankOp(%rank: index) -> index {
  %channel = stream.channel.create rank(%rank) : !stream.channel
  %queried_rank = stream.channel.rank %channel : index
  // CHECK: return %[[RANK]]
  return %queried_rank : index
}

// -----

// CHECK-LABEL: @NoFoldChannelRankOp
func.func @NoFoldChannelRankOp() -> index {
  %channel = stream.channel.create : !stream.channel
  // CHECK: %[[RANK:.+]] = stream.channel.rank
  %queried_rank = stream.channel.rank %channel : index
  // CHECK: return %[[RANK]]
  return %queried_rank : index
}

// -----

// CHECK-LABEL: @FoldChannelCountOp
//  CHECK-SAME: (%[[COUNT:.+]]: index)
func.func @FoldChannelCountOp(%count: index) -> index {
  %channel = stream.channel.create count(%count) : !stream.channel
  %queried_count = stream.channel.count %channel : index
  // CHECK: return %[[COUNT]]
  return %queried_count : index
}

// -----

// CHECK-LABEL: @NoFoldChannelCountOp
func.func @NoFoldChannelCountOp() -> index {
  %channel = stream.channel.create : !stream.channel
  // CHECK: %[[COUNT:.+]] = stream.channel.count
  %queried_count = stream.channel.count %channel : index
  // CHECK: return %[[COUNT]]
  return %queried_count : index
}
