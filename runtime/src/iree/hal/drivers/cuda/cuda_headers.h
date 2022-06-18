// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_CUDA_HEADERS_H_
#define IREE_HAL_DRIVERS_CUDA_CUDA_HEADERS_H_

#include <cuda.h>  // IWYU pragma: export

#ifdef IREE_HAS_NCCL_HEADERS
#include <nccl.h>  // IWYU pragma: export
#endif

#endif  // IREE_HAL_DRIVERS_CUDA_CUDA_HEADERS_H_
