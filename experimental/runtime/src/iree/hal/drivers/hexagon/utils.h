// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_UTILS_H_
#define IREE_HAL_DRIVERS_HEXAGON_UTILS_H_

#include "iree/base/api.h"

/// make an IREE status from a DSP error code
iree_status_t iree_hal_hexagon_make_status_from_dsp_err(const char* file,
                                                        int line, int dsp_err,
                                                        const char* msg);

// make an IREE status from a DSP error code and include current file/line
#define IREE_HAL_HEXAGON_MAKE_STATUS_FROM_DSP_ERR(dsp_err, msg) \
  iree_hal_hexagon_make_status_from_dsp_err(__FILE__, __LINE__, dsp_err, msg)

/// make an IREE status from the errno variable
iree_status_t iree_hal_hexagon_make_status_from_errno(const char* file,
                                                      int line,
                                                      const char* msg);

// make an IREE status from the errno variable and include current file/line
#define IREE_HAL_HEXAGON_MAKE_STATUS_FROM_ERRNO(msg) \
  iree_hal_hexagon_make_status_from_errno(__FILE__, __LINE__, msg)

/// parse a boolean from a string
iree_status_t iree_hal_hexagon_parse_bool_from_string(iree_string_view_t str,
                                                      int* parsed);

#endif  // IREE_HAL_DRIVERS_HEXAGON_UTILS_H_
