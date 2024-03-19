// RUN: iree-opt --split-input-file --canonicalize=test-convergence=true %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldChannelRankOp
//  CHECK-SAME: (%[[RANK:.+]]: index)
util.func private @FoldChannelRankOp(%rank: index) -> index {
  %channel = stream.channel.create rank(%rank) : !stream.channel
  %queried_rank = stream.channel.rank %channel : index
  // CHECK: util.return %[[RANK]]
  util.return %queried_rank : index
}

// -----

// CHECK-LABEL: @NoFoldChannelRankOp
util.func private @NoFoldChannelRankOp() -> index {
  %channel = stream.channel.create : !stream.channel
  // CHECK: %[[RANK:.+]] = stream.channel.rank
  %queried_rank = stream.channel.rank %channel : index
  // CHECK: util.return %[[RANK]]
  util.return %queried_rank : index
}

// -----

// CHECK-LABEL: @FoldChannelCountOp
//  CHECK-SAME: (%[[COUNT:.+]]: index)
util.func private @FoldChannelCountOp(%count: index) -> index {
  %channel = stream.channel.create count(%count) : !stream.channel
  %queried_count = stream.channel.count %channel : index
  // CHECK: util.return %[[COUNT]]
  util.return %queried_count : index
}

// -----

// CHECK-LABEL: @NoFoldChannelCountOp
util.func private @NoFoldChannelCountOp() -> index {
  %channel = stream.channel.create : !stream.channel
  // CHECK: %[[COUNT:.+]] = stream.channel.count
  %queried_count = stream.channel.count %channel : index
  // CHECK: util.return %[[COUNT]]
  util.return %queried_count : index
}
