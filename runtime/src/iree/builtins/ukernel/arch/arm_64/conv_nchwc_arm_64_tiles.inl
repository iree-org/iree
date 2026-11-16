// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Params: ARCH, INPUT_T, FILTER_T, OUTPUT_T, OW_TILE, K0, C0, SUFFIX
//
// The baseline (empty-SUFFIX) entries only use plain NEON and are available
// on every aarch64 CPU.
IREE_UK_CONV_NCHWC_TILE(arm_64, f32, f32, f32, 8, 8, 8, )
