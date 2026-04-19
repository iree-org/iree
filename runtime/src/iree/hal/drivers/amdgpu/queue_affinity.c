// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/queue_affinity.h"

static bool iree_hal_amdgpu_queue_affinity_try_normalize(
    iree_hal_queue_affinity_t supported_affinity,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_normalized_affinity) {
  iree_hal_queue_affinity_t normalized_affinity =
      iree_hal_queue_affinity_is_any(requested_affinity) ? supported_affinity
                                                         : requested_affinity;
  iree_hal_queue_affinity_and_into(normalized_affinity, supported_affinity);
  if (iree_hal_queue_affinity_is_empty(normalized_affinity)) return false;
  *out_normalized_affinity = normalized_affinity;
  return true;
}

static bool iree_hal_amdgpu_queue_affinity_try_resolve_ordinal(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_host_size_t queue_ordinal,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved) {
  if (domain.queue_count_per_physical_device == 0 ||
      queue_ordinal >= IREE_HAL_MAX_QUEUES) {
    return false;
  }

  const iree_host_size_t physical_device_ordinal =
      queue_ordinal / domain.queue_count_per_physical_device;
  if (physical_device_ordinal >= domain.physical_device_count) return false;

  memset(out_resolved, 0, sizeof(*out_resolved));
  out_resolved->queue_affinity = ((iree_hal_queue_affinity_t)1)
                                 << queue_ordinal;
  out_resolved->queue_ordinal = queue_ordinal;
  out_resolved->physical_device_ordinal = physical_device_ordinal;
  out_resolved->physical_queue_ordinal =
      queue_ordinal % domain.queue_count_per_physical_device;
  return true;
}

static bool iree_hal_amdgpu_queue_affinity_try_for_physical_device(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_host_size_t physical_device_ordinal,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  if (domain.queue_count_per_physical_device == 0 ||
      physical_device_ordinal >= domain.physical_device_count) {
    return false;
  }

  iree_host_size_t first_queue_ordinal = 0;
  if (!iree_host_size_checked_mul(physical_device_ordinal,
                                  domain.queue_count_per_physical_device,
                                  &first_queue_ordinal) ||
      first_queue_ordinal >= IREE_HAL_MAX_QUEUES ||
      domain.queue_count_per_physical_device >
          IREE_HAL_MAX_QUEUES - first_queue_ordinal) {
    return false;
  }

  iree_hal_queue_affinity_t queue_affinity = 0;
  for (iree_host_size_t i = 0; i < domain.queue_count_per_physical_device;
       ++i) {
    iree_hal_queue_affinity_or_into(queue_affinity,
                                    ((iree_hal_queue_affinity_t)1)
                                        << (first_queue_ordinal + i));
  }
  *out_queue_affinity = queue_affinity;
  return true;
}

iree_status_t iree_hal_amdgpu_queue_affinity_normalize(
    iree_hal_queue_affinity_t supported_affinity,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_normalized_affinity) {
  *out_normalized_affinity = 0;

  if (!iree_hal_amdgpu_queue_affinity_try_normalize(
          supported_affinity, requested_affinity, out_normalized_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no valid queue affinity bits specified");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_affinity_resolve_ordinal(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_host_size_t queue_ordinal,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved) {
  memset(out_resolved, 0, sizeof(*out_resolved));

  if (IREE_UNLIKELY(domain.queue_count_per_physical_device == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU queue affinity domain has no queues per physical device");
  }
  if (IREE_UNLIKELY(queue_ordinal >= IREE_HAL_MAX_QUEUES)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "queue ordinal %" PRIhsz " exceeds affinity bit capacity %" PRIhsz,
        queue_ordinal, (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  }

  const iree_host_size_t physical_device_ordinal =
      queue_ordinal / domain.queue_count_per_physical_device;
  if (IREE_UNLIKELY(physical_device_ordinal >= domain.physical_device_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "queue ordinal %" PRIhsz
                            " maps to invalid physical device ordinal %" PRIhsz,
                            queue_ordinal, physical_device_ordinal);
  }

  out_resolved->queue_affinity = ((iree_hal_queue_affinity_t)1)
                                 << queue_ordinal;
  out_resolved->queue_ordinal = queue_ordinal;
  out_resolved->physical_device_ordinal = physical_device_ordinal;
  out_resolved->physical_queue_ordinal =
      queue_ordinal % domain.queue_count_per_physical_device;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_affinity_resolve(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved) {
  iree_hal_queue_affinity_t normalized_affinity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_normalize(
      domain.supported_affinity, requested_affinity, &normalized_affinity));

  const iree_host_size_t queue_ordinal =
      iree_hal_queue_affinity_find_first_set(normalized_affinity);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_resolve_ordinal(
      domain, queue_ordinal, out_resolved));
  out_resolved->queue_affinity = normalized_affinity;
  return iree_ok_status();
}

bool iree_hal_amdgpu_queue_affinity_try_resolve(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_amdgpu_queue_affinity_resolved_t* out_resolved) {
  memset(out_resolved, 0, sizeof(*out_resolved));

  iree_hal_queue_affinity_t normalized_affinity = 0;
  if (!iree_hal_amdgpu_queue_affinity_try_normalize(domain.supported_affinity,
                                                    requested_affinity,
                                                    &normalized_affinity)) {
    return false;
  }

  const iree_host_size_t queue_ordinal =
      iree_hal_queue_affinity_find_first_set(normalized_affinity);
  if (!iree_hal_amdgpu_queue_affinity_try_resolve_ordinal(domain, queue_ordinal,
                                                          out_resolved)) {
    return false;
  }
  out_resolved->queue_affinity = normalized_affinity;
  return true;
}

iree_status_t iree_hal_amdgpu_queue_affinity_for_physical_device(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_host_size_t physical_device_ordinal,
    iree_hal_queue_affinity_t* out_queue_affinity) {
  *out_queue_affinity = 0;

  if (IREE_UNLIKELY(domain.queue_count_per_physical_device == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU queue affinity domain has no queues per physical device");
  }
  if (IREE_UNLIKELY(physical_device_ordinal >= domain.physical_device_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "physical device ordinal %" PRIhsz
                            " exceeds physical device count %" PRIhsz,
                            physical_device_ordinal,
                            domain.physical_device_count);
  }

  iree_host_size_t first_queue_ordinal = 0;
  if (!iree_host_size_checked_mul(physical_device_ordinal,
                                  domain.queue_count_per_physical_device,
                                  &first_queue_ordinal) ||
      first_queue_ordinal >= IREE_HAL_MAX_QUEUES ||
      domain.queue_count_per_physical_device >
          IREE_HAL_MAX_QUEUES - first_queue_ordinal) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "physical device queue range does not fit in queue affinity "
        "(physical_device_ordinal=%" PRIhsz
        ", queue_count_per_physical_device=%" PRIhsz ")",
        physical_device_ordinal, domain.queue_count_per_physical_device);
  }

  iree_hal_queue_affinity_t queue_affinity = 0;
  for (iree_host_size_t i = 0; i < domain.queue_count_per_physical_device;
       ++i) {
    iree_hal_queue_affinity_or_into(queue_affinity,
                                    ((iree_hal_queue_affinity_t)1)
                                        << (first_queue_ordinal + i));
  }
  *out_queue_affinity = queue_affinity;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_affinity_select_physical_devices(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_amdgpu_queue_affinity_physical_device_set_t*
        out_physical_device_set) {
  memset(out_physical_device_set, 0, sizeof(*out_physical_device_set));

  if (IREE_UNLIKELY(domain.physical_device_count > IREE_HAL_MAX_QUEUES)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU physical device count %" PRIhsz
                            " exceeds physical device mask capacity %" PRIhsz,
                            domain.physical_device_count,
                            (iree_host_size_t)IREE_HAL_MAX_QUEUES);
  }

  iree_hal_queue_affinity_t normalized_affinity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_normalize(
      domain.supported_affinity, requested_affinity, &normalized_affinity));

  out_physical_device_set->queue_affinity = normalized_affinity;
  for (iree_host_size_t physical_device_ordinal = 0;
       physical_device_ordinal < domain.physical_device_count;
       ++physical_device_ordinal) {
    iree_hal_queue_affinity_t physical_device_affinity = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
        domain, physical_device_ordinal, &physical_device_affinity));
    iree_hal_queue_affinity_and_into(physical_device_affinity,
                                     domain.supported_affinity);

    iree_hal_queue_affinity_t selected_affinity = normalized_affinity;
    iree_hal_queue_affinity_and_into(selected_affinity,
                                     physical_device_affinity);
    if (iree_hal_queue_affinity_is_empty(selected_affinity)) continue;

    if (out_physical_device_set->physical_device_count == 0) {
      out_physical_device_set->first_physical_device_ordinal =
          physical_device_ordinal;
    }
    out_physical_device_set->physical_device_mask |= ((uint64_t)1)
                                                     << physical_device_ordinal;
    ++out_physical_device_set->physical_device_count;
  }

  if (IREE_UNLIKELY(out_physical_device_set->physical_device_count == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue affinity 0x%" PRIx64
                            " selects no physical devices",
                            requested_affinity);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_queue_affinity_normalize_for_physical_device(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_hal_queue_affinity_t* out_queue_affinity,
    iree_host_size_t* out_physical_device_ordinal) {
  *out_queue_affinity = 0;
  *out_physical_device_ordinal = 0;

  iree_hal_amdgpu_queue_affinity_resolved_t resolved;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_resolve(
      domain, requested_affinity, &resolved));

  iree_hal_queue_affinity_t physical_device_affinity = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_queue_affinity_for_physical_device(
      domain, resolved.physical_device_ordinal, &physical_device_affinity));
  iree_hal_queue_affinity_and_into(physical_device_affinity,
                                   domain.supported_affinity);

  const bool is_any_affinity =
      iree_hal_queue_affinity_is_any(requested_affinity);
  if (!is_any_affinity &&
      iree_any_bit_set(resolved.queue_affinity, ~physical_device_affinity)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "AMDGPU queue affinity 0x%" PRIx64
                            " spans multiple physical devices",
                            requested_affinity);
  }

  iree_hal_queue_affinity_t selected_affinity = physical_device_affinity;
  if (!is_any_affinity) {
    selected_affinity = resolved.queue_affinity;
    iree_hal_queue_affinity_and_into(selected_affinity,
                                     physical_device_affinity);
  }

  *out_queue_affinity = selected_affinity;
  *out_physical_device_ordinal = resolved.physical_device_ordinal;
  return iree_ok_status();
}

bool iree_hal_amdgpu_queue_affinity_is_physical_device_local(
    iree_hal_amdgpu_queue_affinity_domain_t domain,
    iree_hal_queue_affinity_t requested_affinity,
    iree_host_size_t physical_device_ordinal) {
  iree_hal_queue_affinity_t normalized_affinity = 0;
  if (!iree_hal_amdgpu_queue_affinity_try_normalize(domain.supported_affinity,
                                                    requested_affinity,
                                                    &normalized_affinity)) {
    return false;
  }

  iree_hal_queue_affinity_t physical_device_affinity = 0;
  if (!iree_hal_amdgpu_queue_affinity_try_for_physical_device(
          domain, physical_device_ordinal, &physical_device_affinity)) {
    return false;
  }
  iree_hal_queue_affinity_and_into(physical_device_affinity,
                                   domain.supported_affinity);
  return !iree_any_bit_set(normalized_affinity, ~physical_device_affinity);
}
