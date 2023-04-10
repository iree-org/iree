// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_UCUDAKERNELS_CONTRACT_H_
#define IREE_COMPILER_CODEGEN_UCUDAKERNELS_CONTRACT_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

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

  uGPUKernel(std::string lhs, std::string rhs, std::string result,
             std::array<INT, 3> tileSize, std::array<INT, 2> warpShape,
             std::array<INT, 3> instShape, INT numStages, bool hasFill,
             bool writeback2Global) {
    LHS = lhs;
    RHS = rhs;
    RESULT = result;
    ThreadblockShape[0] = tileSize[0];
    ThreadblockShape[1] = tileSize[1];
    ThreadblockShape[2] = tileSize[2];
    WarpShape[0] = warpShape[0];
    WarpShape[1] = warpShape[1];
    InstShape[0] = instShape[0];
    InstShape[1] = instShape[1];
    InstShape[2] = instShape[2];
    stages = numStages;
    has_linalg_fill = hasFill;
    writeback_to_global = writeback2Global;
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

static std::string generate_ukernel_name(std::string LHS, std::string RHS,
                                         std::string RES, int TILE_M,
                                         INT TILE_N, INT TILE_K, INT numStages,
                                         bool hasFill, bool writeback2Global) {
  if (LHS == RHS && LHS == "float") {
    uGPUKernel ukernel(LHS, RHS, RES, {TILE_M, TILE_N, TILE_K}, {64, 64},
                       {16, 8, 8}, numStages, hasFill, writeback2Global);
    return ukernel.generate_ukernel_name();
  } else if (LHS == RHS && LHS == "tf32") {
    uGPUKernel ukernel(LHS, RHS, RES, {TILE_M, TILE_N, TILE_K}, {64, 64},
                       {16, 8, 8}, numStages, hasFill, writeback2Global);
    return ukernel.generate_ukernel_name();
  }
  // todo(guray) not supported types
  assert(true);
  return "";
}

/// This is responsible for generating microkernels contracts. The contracts are
/// used in two places. First uCUDAKernelGenerator.cpp (microkernel generator)
/// uses for generatation. Second, it is used in the compiler to make sure we
/// have precompiled microkernel.
struct uGPUContracts {
  std::vector<uGPUKernel> ukernels;

  void generateVariant(std::string lhs, std::string rhs, std::string res,
                       std::array<INT, 3> ts, std::array<INT, 2> ws,
                       std::array<INT, 3> is, INT s) {
    ukernels.push_back(uGPUKernel(lhs, rhs, res, ts, ws, is, s, false, false));
    ukernels.push_back(uGPUKernel(lhs, rhs, res, ts, ws, is, s, true, true));
    ukernels.push_back(uGPUKernel(lhs, rhs, res, ts, ws, is, s, false, true));
    ukernels.push_back(uGPUKernel(lhs, rhs, res, ts, ws, is, s, true, false));
  }

  uGPUContracts() {
    ukernels.clear();

    auto generateF32Microkernels = [&](const char* t, const char* OutT) {
      int st = 2;
      generateVariant(t, t, OutT, {128, 128, 32}, {64, 64}, {16, 8, 8}, st);
      st = 3;
      generateVariant(t, t, OutT, {128, 128, 16}, {64, 64}, {16, 8, 8}, st);
      generateVariant(t, t, OutT, {64, 64, 32}, {64, 64}, {16, 8, 8}, st);
      generateVariant(t, t, OutT, {128, 128, 32}, {64, 64}, {16, 8, 8}, st);
      generateVariant(t, t, OutT, {128, 256, 32}, {64, 64}, {16, 8, 8}, st);
      generateVariant(t, t, OutT, {256, 128, 32}, {64, 64}, {16, 8, 8}, st);
      st = 5;
      generateVariant(t, t, OutT, {128, 256, 16}, {64, 64}, {16, 8, 8}, st);
      generateVariant(t, t, OutT, {128, 128, 16}, {64, 64}, {16, 8, 8}, st);
    };
    // Type float + float
    generateF32Microkernels("float", "float");
    // Type tfloat32 + float
    generateF32Microkernels("tf32", "float");
  }
};

static uGPUContracts AllContracts;

template <class AddTile>
bool adduCUDAContracts(INT M, INT N, INT K, std::string lhs, std::string result,
                       AddTile adder) {
  bool found = false;
  // todo(guray) improve here
  for (uGPUKernel kernel : AllContracts.ukernels) {
    if (kernel.LHS == lhs && kernel.RHS == lhs && kernel.RESULT == result) {
      // Tile size must be bigger than the sizes
      if (M < kernel.ThreadblockShape[0] || N < kernel.ThreadblockShape[1] ||
          K < kernel.ThreadblockShape[2])
        continue;
      // Only divisable tiles for now
      if (M % kernel.ThreadblockShape[0] != 0 ||
          N % kernel.ThreadblockShape[1] != 0 ||
          K % kernel.ThreadblockShape[1] != 0)
        continue;
      // Warps should cover the tiles
      if ((kernel.ThreadblockShape[0] * kernel.ThreadblockShape[1]) %
              (kernel.WarpShape[0] * kernel.WarpShape[1]) !=
          0)
        continue;

      // todo(guray) workaround to not add 4 duplicates
      if (!kernel.has_linalg_fill || !kernel.writeback_to_global) continue;

      int nWarp = (kernel.ThreadblockShape[0] * kernel.ThreadblockShape[1]) /
                  (kernel.WarpShape[0] * kernel.WarpShape[1]);
      adder(kernel.ThreadblockShape[0], kernel.ThreadblockShape[1],
            kernel.ThreadblockShape[2], nWarp, kernel.stages);
      found = true;
    }
  }
  return found;
}

inline bool existuCUDAKernel(INT Tile_M, INT Tile_N, INT Tile_K, INT stages,
                             std::string lhs, std::string rhs,
                             std::string result) {
  // todo(guray) improve here
  for (uGPUKernel kernel : AllContracts.ukernels) {
    if (kernel.LHS == lhs && kernel.RHS == rhs && kernel.RESULT == result &&
        kernel.ThreadblockShape[0] == Tile_M &&
        kernel.ThreadblockShape[1] == Tile_N &&
        kernel.ThreadblockShape[2] == Tile_K && kernel.stages == stages) {
      return true;
    }
  }
  return false;
}

#endif  // IREE_COMPILER_CODEGEN_UKERNELS_UGPUCONTRACT_H_