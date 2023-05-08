// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_PARTITIONER_GSPMD_PIPELINE_H
#define OPENXLA_PARTITIONER_GSPMD_PIPELINE_H

#include "mlir/Pass/PassManager.h"
#include "openxla/partitioner/Support/OptionUtils.h"

namespace openxla::partitioner {

struct GSPMDOptions {
  // The number of partitions to shard by.
  int numPartitions = 1;

  void bindOptions(support::OptionsBinder &binder);
  using FromFlags = support::OptionsFromFlags<GSPMDOptions>;
};

// Builds a pipeline which runs the GSPMD partitioner.
void buildGSPMDPipeline(mlir::PassManager &passManager,
                        const GSPMDOptions &options);

}  // namespace openxla::partitioner

#endif  // OPENXLA_PARTITIONER_GSPMD_PIPELINE_H
