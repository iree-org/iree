// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UCUDAKERNELS_CONTRACT_H_
#define IREE_COMPILER_CODEGEN_UCUDAKERNELS_CONTRACT_H_

#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <set>
#include <string>

#define INT int64_t

inline std::string ugpu_kernel_prefix() { return "__iree_ucuda"; }

inline INT get_cache_line() { return 128; }

struct uGPUKernel {
  std::string LHS, RHS, RESULT;
  std::array<INT, 3> ThreadblockShape;
  std::array<INT, 2> WarpShape;
  std::array<INT, 3> InstShape;
  INT stages;
  bool has_linalg_fill;
  bool writeback_to_global;

  uGPUKernel(std::string lhs, std::string result, std::array<INT, 3> tileSize,
             INT numstages, bool hasFill, bool writeback2Global) {
    // todo(guray) needs to be verification here.
    LHS = RHS = lhs;
    RESULT = result;
    ThreadblockShape[0] = tileSize[0];
    ThreadblockShape[1] = tileSize[1];
    ThreadblockShape[2] = tileSize[2];
    WarpShape = {64, 64};
    InstShape = {16, 8, 8};
    stages = numstages;
    has_linalg_fill = hasFill;
    writeback_to_global = writeback2Global;
  }

  bool operator<(const uGPUKernel& e) const {
    // todo(guray) improve here
    bool result = true;
    if ((ThreadblockShape[0] == e.ThreadblockShape[0] &&
         ThreadblockShape[1] == e.ThreadblockShape[1] &&
         ThreadblockShape[2] == e.ThreadblockShape[2] &&
         has_linalg_fill == e.has_linalg_fill &&
         writeback_to_global == e.writeback_to_global)) {
      result = false;
    }
    return result;
  }

  std::string generate_ukernel_name() {
    std::string ukname;
    ukname.append(ugpu_kernel_prefix() + "_linalg_matmul");
    ukname.append("_");
    ukname.append(LHS);
    ukname.append("_");
    ukname.append(RHS);
    ukname.append("_");
    ukname.append(RESULT);
    for (auto shape : ThreadblockShape) {
      ukname.append("_");
      ukname.append(std::to_string(shape));
    }
    for (auto shape : WarpShape) {
      ukname.append("_");
      ukname.append(std::to_string(shape));
    }
    for (auto shape : InstShape) {
      ukname.append("_");
      ukname.append(std::to_string(shape));
    }
    ukname.append("_");
    ukname.append(std::to_string(stages));
    ukname.append("_");
    ukname.append(has_linalg_fill ? "true" : "false");
    ukname.append("_");
    ukname.append(writeback_to_global ? "true" : "false");
    return ukname;
  }
};

/// This is responsible for generating microkernels contracts. The contracts are
/// used in two places. First uCUDAKernelGenerator.cpp (microkernel generator)
/// uses for generatation. Second, it is used in the compiler to make sure we
/// have precompiled microkernel.
struct uGPUContracts {
  std::set<uGPUKernel> ukernels;

  void generateVariant(std::string lhs, std::string result,
                       std::array<INT, 3> tileSize, INT stages) {
    ukernels.insert(uGPUKernel(lhs, result, tileSize, stages, false, false));
    ukernels.insert(uGPUKernel(lhs, result, tileSize, stages, true, true));
    ukernels.insert(uGPUKernel(lhs, result, tileSize, stages, false, true));
    ukernels.insert(uGPUKernel(lhs, result, tileSize, stages, true, false));
  }

  uGPUContracts() {
    ukernels.clear();
    auto t = "float";
    // Pipeline Stages 3
    generateVariant(t, t, {128, 128, 32}, 3);
    generateVariant(t, t, {128, 256, 32}, 3);
    generateVariant(t, t, {256, 128, 32}, 3);
    // // Pipeline Stages 5
    generateVariant(t, t, {128, 128, 16}, 5);
    generateVariant(t, t, {128, 128, 32}, 5);
  }
};

static uGPUContracts AllContracts;

template <class AddTile>
bool adduCUDAContracts(INT M, INT N, INT K, std::string lhs, std::string result,
                       AddTile adder) {
  bool found = false;
  // todo(guray) improve here
  for (uGPUKernel kernel : AllContracts.ukernels) {
    if (kernel.LHS == lhs && kernel.RHS == lhs && kernel.RESULT == result &&
        M % kernel.ThreadblockShape[0] == 0) {
      int nWarp = (kernel.ThreadblockShape[0] * kernel.ThreadblockShape[1]) /
                  (kernel.WarpShape[0] * kernel.WarpShape[1]);
      adder(kernel.ThreadblockShape[0], kernel.ThreadblockShape[1],
            kernel.ThreadblockShape[2], nWarp, kernel.stages);
      found = true;
    }
  }
  return found;
}

#endif  // IREE_COMPILER_CODEGEN_UKERNELS_UGPUCONTRACT_H_