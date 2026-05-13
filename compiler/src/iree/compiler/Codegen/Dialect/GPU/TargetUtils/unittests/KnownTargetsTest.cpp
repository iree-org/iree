// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>

#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/KnownTargets.h"

namespace mlir::iree_compiler::IREE::GPU {
namespace {

using SmallVec = SmallVector<int64_t>;
using PhaseGroups = SmallVector<SmallVector<int64_t>>;

//===----------------------------------------------------------------------===//
// getSharedMemBankCount
//===----------------------------------------------------------------------===//

TEST(SharedMemoryModel, BankCount32Banks) {
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::CDNA1), 32);
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::CDNA2), 32);
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::CDNA3), 32);
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::RDNA1), 32);
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::RDNA2), 32);
}

TEST(SharedMemoryModel, BankCount64Banks) {
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::CDNA4), 64);
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::RDNA3), 64);
  EXPECT_EQ(getSharedMemBankCount(SharedMemoryModel::RDNA4), 64);
}

//===----------------------------------------------------------------------===//
// getSharedMemBankWidth
//===----------------------------------------------------------------------===//

TEST(SharedMemoryModel, BankWidthAll4Bytes) {
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::CDNA1), 4);
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::CDNA2), 4);
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::CDNA3), 4);
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::CDNA4), 4);
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::RDNA1), 4);
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::RDNA2), 4);
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::RDNA3), 4);
  EXPECT_EQ(getSharedMemBankWidth(SharedMemoryModel::RDNA4), 4);
}

//===----------------------------------------------------------------------===//
// getPhaseGroups — negative cases
//===----------------------------------------------------------------------===//

TEST(PhaseGroupsTest, NonCDNA4ReturnsNullopt) {
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::CDNA1, 16, 64), std::nullopt);
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::CDNA2, 16, 64), std::nullopt);
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::CDNA3, 16, 64), std::nullopt);
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::RDNA1, 16, 64), std::nullopt);
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::RDNA2, 16, 64), std::nullopt);
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::RDNA3, 16, 64), std::nullopt);
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::RDNA4, 16, 64), std::nullopt);
}

TEST(PhaseGroupsTest, CDNA4ReadWidthTooWideReturnsNullopt) {
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::CDNA4, 32, 64), std::nullopt);
}

TEST(PhaseGroupsTest, CDNA4B128WrongThreadCountReturnsNullopt) {
  EXPECT_EQ(getPhaseGroups(SharedMemoryModel::CDNA4, 16, 32), std::nullopt);
}

//===----------------------------------------------------------------------===//
// getPhaseGroups — CDNA4 ds_read_b32 (4-byte reads)
//===----------------------------------------------------------------------===//

TEST(PhaseGroupsTest, CDNA4_B32_Contiguous) {
  // 4-byte read: banksPerAccess=1, threadsPerPhase=64 → 1 phase, all threads.
  auto result = getPhaseGroups(SharedMemoryModel::CDNA4, 4, 64);
  ASSERT_TRUE(result.has_value());
  PhaseGroups &phases = *result;
  ASSERT_EQ(phases.size(), 1u);
  ASSERT_EQ(phases[0].size(), 64u);
  for (int64_t t = 0; t < 64; ++t) {
    EXPECT_EQ(phases[0][t], t);
  }
}

//===----------------------------------------------------------------------===//
// getPhaseGroups — CDNA4 ds_read_b64 (8-byte reads)
//===----------------------------------------------------------------------===//

TEST(PhaseGroupsTest, CDNA4_B64_Contiguous) {
  // 8-byte read: banksPerAccess=2, threadsPerPhase=32 → 2 contiguous phases.
  auto result = getPhaseGroups(SharedMemoryModel::CDNA4, 8, 64);
  ASSERT_TRUE(result.has_value());
  PhaseGroups &phases = *result;
  ASSERT_EQ(phases.size(), 2u);
  ASSERT_EQ(phases[0].size(), 32u);
  ASSERT_EQ(phases[1].size(), 32u);
  for (int64_t t = 0; t < 32; ++t) {
    EXPECT_EQ(phases[0][t], t);
  }
  for (int64_t t = 0; t < 32; ++t) {
    EXPECT_EQ(phases[1][t], t + 32);
  }
}

//===----------------------------------------------------------------------===//
// getPhaseGroups — CDNA4 ds_read_b128 (16-byte reads)
//
// This pins the 64-entry non-contiguous phase table. If any entry in the
// cdna4B128Phases table changes, this test will fail.
//===----------------------------------------------------------------------===//

TEST(PhaseGroupsTest, CDNA4_B128_NonContiguous) {
  auto result = getPhaseGroups(SharedMemoryModel::CDNA4, 16, 64);
  ASSERT_TRUE(result.has_value());
  PhaseGroups &phases = *result;
  ASSERT_EQ(phases.size(), 4u);

  // Phase 0: threads 0-3, 12-15, 20-27
  EXPECT_EQ(phases[0], SmallVec({0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23, 24,
                                 25, 26, 27}));
  // Phase 1: threads 32-35, 44-47, 52-59
  EXPECT_EQ(phases[1], SmallVec({32, 33, 34, 35, 44, 45, 46, 47, 52, 53, 54, 55,
                                 56, 57, 58, 59}));
  // Phase 2: threads 4-11, 16-19, 28-31
  EXPECT_EQ(phases[2], SmallVec({4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 28,
                                 29, 30, 31}));
  // Phase 3: threads 36-43, 48-51, 60-63
  EXPECT_EQ(phases[3], SmallVec({36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 51,
                                 60, 61, 62, 63}));

  // Each phase must have exactly 16 threads.
  for (const auto &phase : phases) {
    EXPECT_EQ(phase.size(), 16u);
  }
}

} // namespace
} // namespace mlir::iree_compiler::IREE::GPU
