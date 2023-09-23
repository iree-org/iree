// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_PARTITIONER_GSPMD_PIPELINE_H
#define OPENXLA_PARTITIONER_GSPMD_PIPELINE_H

#include "absl/container/inlined_vector.h"
#include "mlir/Pass/PassManager.h"
#include "iree_pjrt/partitioner_api/Support/OptionUtils.h"

namespace openxla::partitioner {

struct GSPMDOptions {
  // The number of partitions to shard by.
  int numPartitions = 1;
  // The number of replicas.
  int replicaCount = 1;
  // Whether to use SPMD (true) or MPMD (false) when numPartitions > 0.
  bool useSpmdPartitioning = false;
  // Allows sharding propagation to propagate to the outputs.
  absl::InlinedVector<bool, 1> allowSpmdShardingPropagationToOutput = {false};

  void bindOptions(support::OptionsBinder &binder);
  using FromFlags = support::OptionsFromFlags<GSPMDOptions>;
};

// Builds a pipeline which runs the GSPMD partitioner.
void buildGSPMDPipeline(mlir::PassManager &passManager,
                        const GSPMDOptions &options);

}  // namespace openxla::partitioner

#endif  // OPENXLA_PARTITIONER_GSPMD_PIPELINE_H
