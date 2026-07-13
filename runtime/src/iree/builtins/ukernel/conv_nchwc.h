// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BUILTINS_UKERNEL_CONV_NCHWC_H_
#define IREE_BUILTINS_UKERNEL_CONV_NCHWC_H_

#include "iree/builtins/ukernel/common.h"

// `conv_nchwc` microkernel. Computes a data-tiled NCHWc 2D convolution.
IREE_UK_EXPORT void iree_uk_conv_nchwc(
    const void* input_buffer, iree_uk_index_t input_offset,
    iree_uk_index_t input_stride_n, iree_uk_index_t input_stride_ic_outer,
    iree_uk_index_t input_stride_h, const void* filter_buffer,
    iree_uk_index_t filter_offset, iree_uk_index_t filter_stride_oc_outer,
    iree_uk_index_t filter_stride_ic_outer, iree_uk_index_t filter_stride_fh,
    iree_uk_index_t filter_stride_fw, void* output_buffer,
    iree_uk_index_t output_offset, iree_uk_index_t output_stride_n,
    iree_uk_index_t output_stride_oc_outer, iree_uk_index_t output_stride_oh,
    iree_uk_index_t N, iree_uk_index_t OC_outer, iree_uk_index_t OH,
    iree_uk_index_t OW, iree_uk_index_t IC_outer, iree_uk_index_t FH,
    iree_uk_index_t FW, iree_uk_int32_t k0, iree_uk_int32_t c0,
    iree_uk_int32_t stride_h, iree_uk_int32_t stride_w, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data);

// Returns a bit-field of information about how a conv_nchwc with the given
// tile shape and flags would run on the current target.
IREE_UK_EXPORT iree_uk_uint32_t iree_uk_conv_nchwc_info(
    iree_uk_int32_t k0, iree_uk_int32_t c0, iree_uk_uint32_t flags,
    const iree_uk_uint64_t* cpu_data);

#endif  // IREE_BUILTINS_UKERNEL_CONV_NCHWC_H_
