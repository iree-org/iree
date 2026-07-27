// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Params: ARCH, INPUT_T, FILTER_T, OUTPUT_T, OW_TILE, K0, C0, SUFFIX
//
// AVX-512 provides a single f32 tile config for now:
// OW_TILE=16, K0=16, C0=16
IREE_UK_CONV_NCHWC_TILE(x86_64, f32, f32, f32, 16, 16, 16, _avx512_base)
