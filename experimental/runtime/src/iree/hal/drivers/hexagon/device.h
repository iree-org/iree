// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_HEXAGON_DEVICE_H_
#define IREE_HAL_DRIVERS_HEXAGON_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hexagon/api.h"

//===----------------------------------------------------------------------===//
// Hexagon DSP domains
//===----------------------------------------------------------------------===//

// Returns the name of the DSP domain with |domain_id| in |out_name| (e.g.
// "CDSP"). The name is a string literal and does not need to be freed.
// Returns false if the domain is unknown.
bool iree_hal_hexagon_get_domain_name(iree_hal_hexagon_domain_id_t domain_id,
                                      const char** out_name);

// Returns the ID of the DSP domain with the given |name| (e.g. "CDSP") in
// |out_domain_id|. Returns false if the domain is unknown.
bool iree_hal_hexagon_get_domain_id(
    iree_string_view_t name, iree_hal_hexagon_domain_id_t* out_domain_id);

// Converts between DSP domain IDs and HAL device IDs. The device ID is offset
// by one as domain ID 0 (ADSP) would otherwise be IREE_HAL_DEVICE_ID_DEFAULT.
static inline iree_hal_device_id_t iree_hal_hexagon_device_id_from_domain_id(
    iree_hal_hexagon_domain_id_t domain_id) {
  return (iree_hal_device_id_t)domain_id + 1;
}
static inline iree_hal_hexagon_domain_id_t
iree_hal_hexagon_domain_id_from_device_id(iree_hal_device_id_t device_id) {
  return (iree_hal_hexagon_domain_id_t)(device_id - 1);
}

// Fills |out_device_info| for the DSP domain with |domain_id|. The name and
// path are the domain name. The strings referenced by the device info are
// literals and do not need to be freed.
void iree_hal_hexagon_fill_device_info(iree_hal_hexagon_domain_id_t domain_id,
                                       iree_hal_device_info_t* out_device_info);

#endif  // IREE_HAL_DRIVERS_HEXAGON_DEVICE_H_
