// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UCUDAKERNELS_CONTRACT_H_
#define IREE_COMPILER_CODEGEN_UCUDAKERNELS_CONTRACT_H_

#include <array>
#include <cstdint>
#include <numeric>
#include <set>
#include <string>

inline std::string ugpu_kernel_prefix() { return "__iree_ucuda"; }

inline int64_t get_cache_line() { return 128; }

struct uGPUKernel {
  std::string LHS, RHS, RESULT;
  std::array<int64_t, 3> ThreadblockShape;
  std::array<int64_t, 2> WarpShape;
  std::array<int64_t, 3> InstShape;
  bool has_linalg_fill;

  uGPUKernel(std::string lhs, std::string result,
             std::array<int64_t, 2> tileSize, bool hasFill) {
    // todo(guray) needs to be verification here.

    LHS = RHS = lhs;
    RESULT = result;
    ThreadblockShape[0] = tileSize[0];
    ThreadblockShape[1] = tileSize[1];
    // todo(guray) improve here
    if (LHS == "float") {
      // Tiling K loop for cache line is good rule of thumb.
      int kTile = get_cache_line() / sizeof(float);
      ThreadblockShape[2] = kTile;
      // todo(guray) Need more shapes here
      WarpShape = {64, 64};
      // todo(guray) Need more shapes here
      InstShape = {16, 8, 8};
    }
    has_linalg_fill = hasFill;
  }

  bool operator<(const uGPUKernel& e) const {
    // todo(guray) improve here
    bool result = true;
    if ((ThreadblockShape[0] == e.ThreadblockShape[0] &&
         ThreadblockShape[1] == e.ThreadblockShape[1] &&
         ThreadblockShape[2] == e.ThreadblockShape[2] &&
         has_linalg_fill == e.has_linalg_fill)) {
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
    ukname.append(has_linalg_fill ? "true" : "false");
    return ukname;
  }
};

/// This is responsible for generating microkernels contracts. The contracts are
/// used in two places. First uCUDAKernelGenerator.cpp (microkernel generator)
/// uses for generatation. Second, it is used in the compiler to make sure we
/// have precompiled microkernel.
struct uGPUContracts {
  std::set<uGPUKernel> ukernels;

  uGPUContracts() {
    ukernels.clear();
    ukernels.insert(uGPUKernel("float", "float", {128, 128}, true));

    // todo(guray) Generate more microkernels
    // ukernels.insert(uGPUKernel("float", "float", {128, 128}, true));
    // ukernels.insert(uGPUKernel("float", "float", {64, 64}, true));
    // ukernels.insert(uGPUKernel("float", "float", {64, 64}, false));
  }

  bool is_exist(std::string lhs, std::string result,
                std::array<int64_t, 2> tileSize, bool hasFill) {
    uGPUKernel ukernel(lhs, result, tileSize, hasFill);
    return ukernels.find(ukernel) != ukernels.end();
  }
};

static uGPUContracts AllContracts;

template <class AddTile>
bool adduCUDAContracts(int64_t M, int64_t N, int64_t K, std::string lhs,
                       std::string result, AddTile adder) {
  bool found = false;
  // todo(guray) improve here
  for (uGPUKernel kernel : AllContracts.ukernels) {
    if (kernel.LHS == lhs && kernel.RHS == lhs && kernel.RESULT == result &&
        M % kernel.ThreadblockShape[0] == 0) {
      int nWarp = (kernel.ThreadblockShape[0] * kernel.ThreadblockShape[1]) /
                  (kernel.WarpShape[0] * kernel.WarpShape[1]);
      adder(kernel.ThreadblockShape[0], kernel.ThreadblockShape[1],
            kernel.ThreadblockShape[2], nWarp);
      found = true;
    }
  }
  return found;
}

static bool hasuCUDAContract(std::string lhs, std::string result,
                             std::array<int64_t, 2> tileSize, bool hasFill) {
  return AllContracts.is_exist(lhs, result, tileSize, hasFill);
}

#endif  // IREE_COMPILER_CODEGEN_UKERNELS_UGPUCONTRACT_H_