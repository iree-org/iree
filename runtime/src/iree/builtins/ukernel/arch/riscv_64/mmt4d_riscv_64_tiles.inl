// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordering matters when multiple lines have the same types and tile shape and
// are supported by the CPU. In that case, the last-enumerated line overrides
// // preceding lines. Always go from oldest to shiniest code path.

IREE_UK_MMT4D_TILE(riscv_64, f32, f32, f32, 1, 1, _v)
IREE_UK_MMT4D_TILE(riscv_64, f32, f32, f32, 2, 1, _v)
IREE_UK_MMT4D_TILE(riscv_64, f32, f32, f32, 4, 1, _v)
IREE_UK_MMT4D_TILE(riscv_64, f32, f32, f32, 7, 1, _v)
