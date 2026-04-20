// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_device_metrics.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/alignment.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"

#if defined(IREE_PLATFORM_LINUX)
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_LINUX

//===----------------------------------------------------------------------===//
// Linux sysfs/gpu_metrics schema subset
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_PATH_LENGTH 256
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_NAME_LENGTH 64
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_SAMPLE_VALUES 16
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_GPU_BUFFER_LENGTH 4096
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_TEXT_BUFFER_LENGTH 64
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_SOURCE_KIND_SYSFS 1u

typedef char iree_hal_amdgpu_profile_device_metrics_text_buffer_t
    [IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_TEXT_BUFFER_LENGTH];

// Built-in metrics emitted by the AMDGPU sysfs sampler.
static const uint64_t iree_hal_amdgpu_profile_device_metrics_metric_ids[] = {
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_COMPUTE_CURRENT,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_MEMORY_CURRENT,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_EDGE,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_HOTSPOT,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_MEMORY,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_POWER_SOCKET,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_COMPUTE,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_MEMORY,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_MEMORY_LOCAL_USED,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_MEMORY_LOCAL_TOTAL,
    IREE_HAL_PROFILE_BUILTIN_METRIC_ID_THROTTLE_STATUS,
};

typedef struct iree_hal_amdgpu_profile_metrics_table_header_t {
  // Total byte size of the returned table.
  uint16_t structure_size;
  // Major gpu_metrics layout family.
  uint8_t format_revision;
  // Minor gpu_metrics layout revision within |format_revision|.
  uint8_t content_revision;
} iree_hal_amdgpu_profile_metrics_table_header_t;

// Naturally aligned subset matching Linux gpu_metrics_v1_1 through v1_3.
typedef struct iree_hal_amdgpu_profile_gpu_metrics_v1_3_t {
  // Shared metrics table header.
  iree_hal_amdgpu_profile_metrics_table_header_t common_header;
  // Edge temperature in degrees Celsius.
  uint16_t temperature_edge;
  // Hotspot temperature in degrees Celsius.
  uint16_t temperature_hotspot;
  // Memory temperature in degrees Celsius.
  uint16_t temperature_mem;
  // VR gfx temperature in degrees Celsius.
  uint16_t temperature_vrgfx;
  // VR SoC temperature in degrees Celsius.
  uint16_t temperature_vrsoc;
  // VR memory temperature in degrees Celsius.
  uint16_t temperature_vrmem;
  // Average gfx activity percentage.
  uint16_t average_gfx_activity;
  // Average memory-controller activity percentage.
  uint16_t average_umc_activity;
  // Average media activity percentage.
  uint16_t average_mm_activity;
  // Average socket power in Watts.
  uint16_t average_socket_power;
  // Producer-defined energy accumulator.
  uint64_t energy_accumulator;
  // Driver-attached timestamp in nanoseconds.
  uint64_t system_clock_counter;
  // Average gfx clock in MHz.
  uint16_t average_gfxclk_frequency;
  // Average SoC clock in MHz.
  uint16_t average_socclk_frequency;
  // Average memory clock in MHz.
  uint16_t average_uclk_frequency;
  // Average VCLK0 clock in MHz.
  uint16_t average_vclk0_frequency;
  // Average DCLK0 clock in MHz.
  uint16_t average_dclk0_frequency;
  // Average VCLK1 clock in MHz.
  uint16_t average_vclk1_frequency;
  // Average DCLK1 clock in MHz.
  uint16_t average_dclk1_frequency;
  // Current gfx clock in MHz.
  uint16_t current_gfxclk;
  // Current SoC clock in MHz.
  uint16_t current_socclk;
  // Current memory clock in MHz.
  uint16_t current_uclk;
  // Current VCLK0 clock in MHz.
  uint16_t current_vclk0;
  // Current DCLK0 clock in MHz.
  uint16_t current_dclk0;
  // Current VCLK1 clock in MHz.
  uint16_t current_vclk1;
  // Current DCLK1 clock in MHz.
  uint16_t current_dclk1;
  // ASIC-dependent throttle status bitfield.
  uint32_t throttle_status;
  // Current fan speed in RPM.
  uint16_t current_fan_speed;
  // Current PCIe link width.
  uint16_t pcie_link_width;
  // Current PCIe link speed in tenths of GT/s.
  uint16_t pcie_link_speed;
  // Padding matching the kernel ABI.
  uint16_t padding;
  // Accumulated gfx activity.
  uint32_t gfx_activity_acc;
  // Accumulated memory activity.
  uint32_t mem_activity_acc;
  // HBM instance temperatures in degrees Celsius.
  uint16_t temperature_hbm[4];
  // PMFW timestamp with 10ns resolution.
  uint64_t firmware_timestamp;
  // SoC voltage in mV.
  uint16_t voltage_soc;
  // Gfx voltage in mV.
  uint16_t voltage_gfx;
  // Memory voltage in mV.
  uint16_t voltage_mem;
  // Padding matching the kernel ABI.
  uint16_t padding1;
  // ASIC-independent throttle status bitfield.
  uint64_t indep_throttle_status;
} iree_hal_amdgpu_profile_gpu_metrics_v1_3_t;

// Scalar sysfs metric slots discovered once at profiling begin.
typedef uint32_t iree_hal_amdgpu_profile_scalar_metric_slot_t;
enum iree_hal_amdgpu_profile_scalar_metric_slot_e {
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_GPU_BUSY_PERCENT = 0,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_BUSY_PERCENT = 1,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_LOCAL_USED = 2,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_LOCAL_TOTAL = 3,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_CLOCK_COMPUTE_CURRENT = 4,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_CLOCK_MEMORY_CURRENT = 5,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_EDGE = 6,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_HOTSPOT = 7,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_MEMORY = 8,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_POWER_SOCKET = 9,
  IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_COUNT = 10,
};

typedef struct iree_hal_amdgpu_profile_scalar_metric_t {
  // Built-in metric id emitted for this scalar file.
  uint64_t metric_id;
  // Multiplier applied to the parsed unsigned sysfs value.
  uint64_t scale;
} iree_hal_amdgpu_profile_scalar_metric_t;

static iree_hal_amdgpu_profile_scalar_metric_t
iree_hal_amdgpu_profile_device_metrics_scalar_metric(
    iree_hal_amdgpu_profile_scalar_metric_slot_t slot) {
  switch (slot) {
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_GPU_BUSY_PERCENT:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_COMPUTE,
          .scale = 1000u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_BUSY_PERCENT:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_MEMORY,
          .scale = 1000u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_LOCAL_USED:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_MEMORY_LOCAL_USED,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_LOCAL_TOTAL:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_MEMORY_LOCAL_TOTAL,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_CLOCK_COMPUTE_CURRENT:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_COMPUTE_CURRENT,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_CLOCK_MEMORY_CURRENT:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_MEMORY_CURRENT,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_EDGE:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_EDGE,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_HOTSPOT:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_HOTSPOT,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_MEMORY:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_MEMORY,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_POWER_SOCKET:
      return (iree_hal_amdgpu_profile_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_POWER_SOCKET,
          .scale = 1u,
      };
    default:
      return (iree_hal_amdgpu_profile_scalar_metric_t){0};
  }
}

typedef struct iree_hal_amdgpu_profile_device_metric_source_t {
  // Producer-defined source id unique within the profiling session.
  uint64_t source_id;
  // Next nonzero sample id emitted for this source.
  uint64_t next_sample_id;
  // Session-local physical device ordinal sampled by this source.
  uint32_t physical_device_ordinal;
  // Number of readable sysfs files discovered for this source.
  uint32_t readable_file_count;
  // Source revision derived from the gpu_metrics header when readable.
  uint32_t source_revision;
  // Human-readable source name stored in source metadata.
  char name[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_NAME_LENGTH];
  // NUL-terminated sysfs PCI device path.
  char device_path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_PATH_LENGTH];
  // Open gpu_metrics binary sysfs file descriptor, or -1.
  int gpu_metrics_file_descriptor;
  // Open scalar sysfs file descriptors indexed by metric slot, or -1.
  int scalar_file_descriptors[IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_COUNT];
} iree_hal_amdgpu_profile_device_metric_source_t;

static int* iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_amdgpu_profile_scalar_metric_slot_t slot) {
  return &source->scalar_file_descriptors[slot];
}

struct iree_hal_amdgpu_profile_device_metrics_session_t {
  // Host allocator used for session storage.
  iree_allocator_t host_allocator;
  // Number of initialized entries in |sources|.
  iree_host_size_t source_count;
  // Per-physical-device metric sources.
  iree_hal_amdgpu_profile_device_metric_source_t sources[];
};

typedef struct iree_hal_amdgpu_profile_device_metric_sample_builder_t {
  // Sample record header being populated.
  iree_hal_profile_device_metric_sample_record_t record;
  // Fixed value storage written immediately after |record|.
  iree_hal_profile_device_metric_value_t
      values[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_SAMPLE_VALUES];
} iree_hal_amdgpu_profile_device_metric_sample_builder_t;

//===----------------------------------------------------------------------===//
// File discovery and sampling utilities
//===----------------------------------------------------------------------===//

static bool iree_hal_amdgpu_profile_device_metrics_u16_is_valid(
    uint16_t value) {
  return value != UINT16_MAX;
}

static bool iree_hal_amdgpu_profile_device_metrics_u32_is_valid(
    uint32_t value) {
  return value != UINT32_MAX;
}

static bool iree_hal_amdgpu_profile_device_metrics_u64_is_valid(
    uint64_t value) {
  return value != UINT64_MAX;
}

static bool iree_hal_amdgpu_profile_device_metrics_value_is_present(
    const iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id) {
  for (uint32_t i = 0; i < builder->record.value_count; ++i) {
    if (builder->values[i].metric_id == metric_id) return true;
  }
  return false;
}

static bool iree_hal_amdgpu_profile_device_metrics_has_field(
    iree_host_size_t storage_length, iree_host_size_t field_offset,
    iree_host_size_t field_size) {
  return field_offset <= storage_length &&
         field_size <= storage_length - field_offset;
}

#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(storage_length, type, \
                                                         field)                \
  iree_hal_amdgpu_profile_device_metrics_has_field(                            \
      (storage_length), offsetof(type, field),                                 \
      sizeof(((const type*)0)->field))

static void iree_hal_amdgpu_profile_device_metrics_append_u64(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, uint64_t value) {
  if (iree_hal_amdgpu_profile_device_metrics_value_is_present(builder,
                                                              metric_id)) {
    return;
  }
  if (builder->record.value_count >=
      IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_SAMPLE_VALUES) {
    builder->record.flags |= IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
    return;
  }
  builder->values[builder->record.value_count++] =
      (iree_hal_profile_device_metric_value_t){
          .metric_id = metric_id,
          .value_bits = value,
      };
}

static void iree_hal_amdgpu_profile_device_metrics_append_i64(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, int64_t value) {
  iree_hal_amdgpu_profile_device_metrics_append_u64(builder, metric_id,
                                                    (uint64_t)value);
}

static void iree_hal_amdgpu_profile_device_metrics_append_u16_scaled(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, uint16_t value, uint64_t scale) {
  if (iree_hal_amdgpu_profile_device_metrics_u16_is_valid(value)) {
    iree_hal_amdgpu_profile_device_metrics_append_u64(builder, metric_id,
                                                      (uint64_t)value * scale);
  }
}

static void iree_hal_amdgpu_profile_device_metrics_append_i16_scaled(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, uint16_t value, int64_t scale) {
  if (iree_hal_amdgpu_profile_device_metrics_u16_is_valid(value)) {
    iree_hal_amdgpu_profile_device_metrics_append_i64(builder, metric_id,
                                                      (int64_t)value * scale);
  }
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_format_path(
    char* buffer, iree_host_size_t buffer_capacity, const char* format, ...) {
  va_list varargs;
  va_start(varargs, format);
  const int result = vsnprintf(buffer, buffer_capacity, format, varargs);
  va_end(varargs);
  if (result < 0 || (iree_host_size_t)result >= buffer_capacity) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU metric sysfs path exceeds %" PRIhsz " bytes", buffer_capacity);
  }
  return iree_ok_status();
}

#if defined(IREE_PLATFORM_LINUX)

static bool iree_hal_amdgpu_profile_device_metrics_is_optional_file_error(
    int error_code) {
  return error_code == ENOENT || error_code == ENOTDIR || error_code == ENODEV;
}

static bool iree_hal_amdgpu_profile_device_metrics_is_unavailable_read_error(
    int error_code) {
  return error_code == EBUSY || error_code == EAGAIN || error_code == ENODATA ||
         error_code == ENODEV;
}

static void iree_hal_amdgpu_profile_device_metrics_close_file(
    int* file_descriptor) {
  if (*file_descriptor >= 0) {
    close(*file_descriptor);
    *file_descriptor = -1;
  }
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_open_optional_file(
    const char* path, int* out_file_descriptor) {
  *out_file_descriptor = -1;
  const int file_descriptor = open(path, O_RDONLY | O_CLOEXEC);
  if (file_descriptor >= 0) {
    *out_file_descriptor = file_descriptor;
    return iree_ok_status();
  }

  const int error_code = errno;
  if (iree_hal_amdgpu_profile_device_metrics_is_optional_file_error(
          error_code)) {
    return iree_ok_status();
  }
  return iree_make_status(iree_status_code_from_errno(error_code),
                          "failed to open AMDGPU metric sysfs file %s: %s",
                          path, strerror(error_code));
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_read_file(
    int file_descriptor, const char* name, uint8_t* buffer,
    iree_host_size_t buffer_capacity, bool* out_available,
    iree_host_size_t* out_length) {
  *out_available = false;
  *out_length = 0;
  if (file_descriptor < 0) return iree_ok_status();

  ssize_t read_length = 0;
  do {
    read_length = pread(file_descriptor, buffer, buffer_capacity, 0);
  } while (read_length < 0 && errno == EINTR);
  if (read_length < 0) {
    const int error_code = errno;
    if (iree_hal_amdgpu_profile_device_metrics_is_unavailable_read_error(
            error_code)) {
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error_code),
                            "failed to read AMDGPU metric sysfs file %s: %s",
                            name, strerror(error_code));
  }
  if (read_length == 0) return iree_ok_status();

  *out_available = true;
  *out_length = (iree_host_size_t)read_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_read_uint64_file(
    int file_descriptor, const char* name, bool* out_available,
    uint64_t* out_value) {
  *out_available = false;
  *out_value = 0;
  iree_hal_amdgpu_profile_device_metrics_text_buffer_t buffer = {0};
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_read_file(
      file_descriptor, name, buffer, sizeof(buffer) - 1, out_available,
      &length));
  if (!*out_available) return iree_ok_status();

  buffer[length] = 0;
  iree_string_view_t text =
      iree_string_view_trim(iree_make_string_view((const char*)buffer, length));
  if (!iree_string_view_atoi_uint64(text, out_value)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "failed to parse AMDGPU metric sysfs file %s as "
                            "uint64: %.*s",
                            name, (int)text.size, text.data);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_open_device_file(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    const char* file_name, int* out_file_descriptor) {
  char path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_PATH_LENGTH] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      path, sizeof(path), "%s/%s", source->device_path, file_name));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_open_optional_file(
          path, out_file_descriptor));
  if (*out_file_descriptor >= 0) ++source->readable_file_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_read_label(
    const char* directory_path, const char* prefix, uint32_t index,
    char* buffer, iree_host_size_t buffer_capacity, bool* out_available,
    iree_string_view_t* out_label) {
  *out_label = iree_string_view_empty();
  char path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_PATH_LENGTH] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      path, sizeof(path), "%s/%s%" PRIu32 "_label", directory_path, prefix,
      index));
  int file_descriptor = -1;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_open_optional_file(
          path, &file_descriptor));
  iree_host_size_t length = 0;
  iree_status_t status = iree_hal_amdgpu_profile_device_metrics_read_file(
      file_descriptor, path, (uint8_t*)buffer, buffer_capacity - 1,
      out_available, &length);
  iree_hal_amdgpu_profile_device_metrics_close_file(&file_descriptor);
  if (!iree_status_is_ok(status) || !*out_available) return status;

  buffer[length] = 0;
  *out_label =
      iree_string_view_trim(iree_make_string_view((const char*)buffer, length));
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_amdgpu_profile_scalar_metric_slot_t slot,
    const char* directory_path, const char* prefix, uint32_t index,
    const char* suffix) {
  if (source->scalar_file_descriptors[slot] >= 0) return iree_ok_status();
  char path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_PATH_LENGTH] = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      path, sizeof(path), "%s/%s%" PRIu32 "_%s", directory_path, prefix, index,
      suffix));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_open_optional_file(
          path, &source->scalar_file_descriptors[slot]));
  if (source->scalar_file_descriptors[slot] >= 0) ++source->readable_file_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_discover_hwmon_freq(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    const char* directory_path) {
  for (uint32_t i = 1; i <= 8; ++i) {
    iree_hal_amdgpu_profile_device_metrics_text_buffer_t label_buffer;
    bool label_available = false;
    iree_string_view_t label = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_read_label(
        directory_path, "freq", i, label_buffer, sizeof(label_buffer),
        &label_available, &label));
    if (!label_available) continue;

    if (iree_string_view_equal(label, IREE_SV("sclk")) ||
        iree_string_view_equal(label, IREE_SV("gfxclk"))) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
              source,
              IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_CLOCK_COMPUTE_CURRENT,
              directory_path, "freq", i, "input"));
    } else if (iree_string_view_equal(label, IREE_SV("mclk")) ||
               iree_string_view_equal(label, IREE_SV("uclk"))) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
              source,
              IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_CLOCK_MEMORY_CURRENT,
              directory_path, "freq", i, "input"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_discover_hwmon_temperature(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    const char* directory_path) {
  for (uint32_t i = 1; i <= 16; ++i) {
    iree_hal_amdgpu_profile_device_metrics_text_buffer_t label_buffer;
    bool label_available = false;
    iree_string_view_t label = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_read_label(
        directory_path, "temp", i, label_buffer, sizeof(label_buffer),
        &label_available, &label));
    if (!label_available) continue;

    if (iree_string_view_equal(label, IREE_SV("edge"))) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
              source,
              IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_EDGE,
              directory_path, "temp", i, "input"));
    } else if (iree_string_view_equal(label, IREE_SV("junction")) ||
               iree_string_view_equal(label, IREE_SV("hotspot"))) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
              source,
              IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_HOTSPOT,
              directory_path, "temp", i, "input"));
    } else if (iree_string_view_equal(label, IREE_SV("mem")) ||
               iree_string_view_equal(label, IREE_SV("memory"))) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
              source,
              IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_TEMPERATURE_MEMORY,
              directory_path, "temp", i, "input"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_discover_hwmon_power(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    const char* directory_path) {
  return iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
      source, IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_POWER_SOCKET,
      directory_path, "power", 1, "average");
}

static bool iree_hal_amdgpu_profile_device_metrics_is_hwmon_dirent(
    const char* name) {
  if (strncmp(name, "hwmon", 5) != 0) return false;
  for (const char* cursor = name + 5; *cursor; ++cursor) {
    if (*cursor < '0' || *cursor > '9') return false;
  }
  return true;
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_discover_hwmon(
    iree_hal_amdgpu_profile_device_metric_source_t* source) {
  char directory_path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_PATH_LENGTH] =
      {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      directory_path, sizeof(directory_path), "%s/hwmon", source->device_path));

  DIR* directory = opendir(directory_path);
  if (!directory) {
    const int error_code = errno;
    if (iree_hal_amdgpu_profile_device_metrics_is_optional_file_error(
            error_code)) {
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(error_code),
                            "failed to open AMDGPU hwmon directory %s: %s",
                            directory_path, strerror(error_code));
  }

  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    errno = 0;
    struct dirent* entry = readdir(directory);
    if (!entry) {
      if (errno != 0) {
        const int error_code = errno;
        status = iree_make_status(iree_status_code_from_errno(error_code),
                                  "failed to enumerate AMDGPU hwmon directory "
                                  "%s: %s",
                                  directory_path, strerror(error_code));
      }
      break;
    }
    if (!iree_hal_amdgpu_profile_device_metrics_is_hwmon_dirent(
            entry->d_name)) {
      continue;
    }
    char hwmon_path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_MAX_PATH_LENGTH] = {
        0};
    status = iree_hal_amdgpu_profile_device_metrics_format_path(
        hwmon_path, sizeof(hwmon_path), "%s/%s", directory_path, entry->d_name);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_profile_device_metrics_discover_hwmon_freq(
          source, hwmon_path);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_amdgpu_profile_device_metrics_discover_hwmon_temperature(
              source, hwmon_path);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_profile_device_metrics_discover_hwmon_power(
          source, hwmon_path);
    }
  }
  closedir(directory);
  return status;
}

static void iree_hal_amdgpu_profile_device_metrics_parse_gpu_metrics_v1_3(
    const uint8_t* storage, iree_host_size_t storage_length,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  const iree_hal_amdgpu_profile_gpu_metrics_v1_3_t* metrics =
      (const iree_hal_amdgpu_profile_gpu_metrics_v1_3_t*)(const void*)storage;

  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          temperature_edge)) {
    iree_hal_amdgpu_profile_device_metrics_append_i16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_EDGE,
        metrics->temperature_edge, 1000);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          temperature_hotspot)) {
    iree_hal_amdgpu_profile_device_metrics_append_i16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_HOTSPOT,
        metrics->temperature_hotspot, 1000);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          temperature_mem)) {
    iree_hal_amdgpu_profile_device_metrics_append_i16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_MEMORY,
        metrics->temperature_mem, 1000);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          average_gfx_activity)) {
    iree_hal_amdgpu_profile_device_metrics_append_u16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_COMPUTE,
        metrics->average_gfx_activity, 1000u);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          average_umc_activity)) {
    iree_hal_amdgpu_profile_device_metrics_append_u16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_MEMORY,
        metrics->average_umc_activity, 1000u);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          average_socket_power)) {
    iree_hal_amdgpu_profile_device_metrics_append_u16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_POWER_SOCKET,
        metrics->average_socket_power, 1000000u);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          current_gfxclk)) {
    iree_hal_amdgpu_profile_device_metrics_append_u16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_COMPUTE_CURRENT,
        metrics->current_gfxclk, 1000000u);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          current_uclk)) {
    iree_hal_amdgpu_profile_device_metrics_append_u16_scaled(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_MEMORY_CURRENT,
        metrics->current_uclk, 1000000u);
  }
  if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
          storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
          indep_throttle_status) &&
      iree_hal_amdgpu_profile_device_metrics_u64_is_valid(
          metrics->indep_throttle_status)) {
    iree_hal_amdgpu_profile_device_metrics_append_u64(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_THROTTLE_STATUS,
        metrics->indep_throttle_status);
  } else if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
                 storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
                 throttle_status) &&
             iree_hal_amdgpu_profile_device_metrics_u32_is_valid(
                 metrics->throttle_status)) {
    iree_hal_amdgpu_profile_device_metrics_append_u64(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_THROTTLE_STATUS,
        metrics->throttle_status);
  }
}

#undef IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD

static iree_status_t iree_hal_amdgpu_profile_device_metrics_parse_gpu_metrics(
    const uint8_t* storage, iree_host_size_t storage_length,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint32_t* out_source_revision) {
  if (storage_length < sizeof(iree_hal_amdgpu_profile_metrics_table_header_t)) {
    builder->record.flags |= IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
    return iree_ok_status();
  }
  const iree_hal_amdgpu_profile_metrics_table_header_t* header =
      (const iree_hal_amdgpu_profile_metrics_table_header_t*)(const void*)
          storage;
  *out_source_revision =
      ((uint32_t)header->format_revision << 8) | header->content_revision;
  if (header->structure_size <
      sizeof(iree_hal_amdgpu_profile_metrics_table_header_t)) {
    builder->record.flags |= IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
    return iree_ok_status();
  }

  const iree_host_size_t table_length =
      iree_min(storage_length, (iree_host_size_t)header->structure_size);
  if (header->format_revision == 1 && header->content_revision >= 1 &&
      header->content_revision <= 3) {
    iree_hal_amdgpu_profile_device_metrics_parse_gpu_metrics_v1_3(
        storage, table_length, builder);
    return iree_ok_status();
  }

  builder->record.flags |= IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_probe_gpu_metrics(
    iree_hal_amdgpu_profile_device_metric_source_t* source) {
  uint8_t storage[sizeof(iree_hal_amdgpu_profile_metrics_table_header_t)] = {0};
  bool available = false;
  iree_host_size_t storage_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_read_file(
      source->gpu_metrics_file_descriptor, "gpu_metrics", storage,
      sizeof(storage), &available, &storage_length));
  if (!available ||
      storage_length < sizeof(iree_hal_amdgpu_profile_metrics_table_header_t)) {
    return iree_ok_status();
  }

  const iree_hal_amdgpu_profile_metrics_table_header_t* header =
      (const iree_hal_amdgpu_profile_metrics_table_header_t*)(const void*)
          storage;
  source->source_revision =
      ((uint32_t)header->format_revision << 8) | header->content_revision;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_sample_gpu_metrics(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  uint8_t storage[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_GPU_BUFFER_LENGTH] = {
      0};
  bool available = false;
  iree_host_size_t storage_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_read_file(
      source->gpu_metrics_file_descriptor, "gpu_metrics", storage,
      sizeof(storage), &available, &storage_length));
  if (!available) {
    if (source->gpu_metrics_file_descriptor >= 0) {
      builder->record.flags |=
          IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
    }
    return iree_ok_status();
  }
  uint32_t source_revision = source->source_revision;
  return iree_hal_amdgpu_profile_device_metrics_parse_gpu_metrics(
      storage, storage_length, builder, &source_revision);
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_sample_scalars(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  for (iree_host_size_t i = 0;
       i < IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_COUNT; ++i) {
    const int file_descriptor = source->scalar_file_descriptors[i];
    if (file_descriptor < 0) continue;

    bool available = false;
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_profile_device_metrics_read_uint64_file(
            file_descriptor, "scalar metric", &available, &value));
    if (!available) {
      builder->record.flags |=
          IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
      continue;
    }
    const iree_hal_amdgpu_profile_scalar_metric_t scalar_metric =
        iree_hal_amdgpu_profile_device_metrics_scalar_metric(
            (iree_hal_amdgpu_profile_scalar_metric_slot_t)i);
    iree_hal_amdgpu_profile_device_metrics_append_u64(
        builder, scalar_metric.metric_id, value * scalar_metric.scale);
  }
  return iree_ok_status();
}

#else

static void iree_hal_amdgpu_profile_device_metrics_close_file(
    int* file_descriptor) {
  *file_descriptor = -1;
}

#endif  // IREE_PLATFORM_LINUX

//===----------------------------------------------------------------------===//
// Profile record emission
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_profile_device_metrics_write_chunk(
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name, iree_string_view_t content_type,
    const uint8_t* storage, iree_host_size_t storage_size) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.name = stream_name;
  metadata.session_id = session_id;
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span(storage, storage_size);
  return iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_source_record_size(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_host_size_t* out_record_size) {
  const iree_host_size_t name_length = strlen(source->name);
  return IREE_STRUCT_LAYOUT(
      0, out_record_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_device_metric_source_record_t,
                        NULL),
      IREE_STRUCT_FIELD(name_length, char, NULL));
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_descriptor_record_size(
    const iree_hal_profile_metric_descriptor_t* descriptor,
    iree_host_size_t* out_record_size) {
  return IREE_STRUCT_LAYOUT(
      0, out_record_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_device_metric_descriptor_record_t,
                        NULL),
      IREE_STRUCT_FIELD(descriptor->name.size + descriptor->description.size,
                        char, NULL));
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_pack_source_record(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    uint8_t* storage, iree_host_size_t storage_capacity,
    iree_host_size_t* out_storage_size) {
  iree_host_size_t record_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_source_record_size(source,
                                                                &record_size));
  if (record_size > UINT32_MAX || record_size > storage_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metric source record exceeds storage");
  }

  const iree_host_size_t name_length = strlen(source->name);
  iree_hal_profile_device_metric_source_record_t record =
      iree_hal_profile_device_metric_source_record_default();
  record.record_length = (uint32_t)record_size;
  record.source_id = source->source_id;
  record.physical_device_ordinal = source->physical_device_ordinal;
  record.device_class = IREE_HAL_PROFILE_DEVICE_CLASS_GPU;
  record.source_kind = IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_SOURCE_KIND_SYSFS;
  record.source_revision = source->source_revision;
  record.metric_count =
      IREE_ARRAYSIZE(iree_hal_amdgpu_profile_device_metrics_metric_ids);
  record.name_length = (uint32_t)name_length;

  memcpy(storage, &record, sizeof(record));
  memcpy(storage + sizeof(record), source->name, name_length);
  *out_storage_size = record_size;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_pack_descriptor_record(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    const iree_hal_profile_metric_descriptor_t* descriptor, uint8_t* storage,
    iree_host_size_t storage_capacity, iree_host_size_t* out_storage_size) {
  iree_host_size_t record_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_descriptor_record_size(
          descriptor, &record_size));
  if (record_size > UINT32_MAX || record_size > storage_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metric descriptor record exceeds storage");
  }

  iree_hal_profile_device_metric_descriptor_record_t record =
      iree_hal_profile_device_metric_descriptor_record_default();
  record.record_length = (uint32_t)record_size;
  record.source_id = source->source_id;
  record.metric_id = descriptor->metric_id;
  record.unit = descriptor->unit;
  record.value_kind = descriptor->value_kind;
  record.semantic = descriptor->semantic;
  record.plot_hint = descriptor->plot_hint;
  record.name_length = (uint32_t)descriptor->name.size;
  record.description_length = (uint32_t)descriptor->description.size;

  memcpy(storage, &record, sizeof(record));
  uint8_t* string_ptr = storage + sizeof(record);
  memcpy(string_ptr, descriptor->name.data, descriptor->name.size);
  string_ptr += descriptor->name.size;
  memcpy(string_ptr, descriptor->description.data,
         descriptor->description.size);
  *out_storage_size = record_size;
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_write_source_metadata(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  uint8_t storage[128] = {0};
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_pack_source_record(
          source, storage, sizeof(storage), &storage_size));
  return iree_hal_amdgpu_profile_device_metrics_write_chunk(
      sink, session_id, stream_name,
      IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SOURCES, storage,
      storage_size);
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_write_descriptor_metadata(
    const iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name, uint64_t metric_id) {
  const iree_hal_profile_metric_descriptor_t* descriptor =
      iree_hal_profile_builtin_metric_descriptor_lookup(metric_id);
  if (!descriptor) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AMDGPU metric id %" PRIu64 " has no built-in descriptor", metric_id);
  }

  uint8_t storage[256] = {0};
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_pack_descriptor_record(
          source, descriptor, storage, sizeof(storage), &storage_size));
  return iree_hal_amdgpu_profile_device_metrics_write_chunk(
      sink, session_id, stream_name,
      IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_DESCRIPTORS, storage,
      storage_size);
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_write_sample(
    const iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  iree_host_size_t storage_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &storage_size,
      IREE_STRUCT_FIELD(1, iree_hal_profile_device_metric_sample_record_t,
                        NULL),
      IREE_STRUCT_FIELD(builder->record.value_count,
                        iree_hal_profile_device_metric_value_t, NULL)));
  if (IREE_UNLIKELY(storage_size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU metric sample exceeds uint32_t");
  }

  iree_hal_profile_device_metric_sample_record_t record = builder->record;
  record.record_length = (uint32_t)storage_size;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span((const uint8_t*)&record, sizeof(record)),
      iree_make_const_byte_span(
          (const uint8_t*)builder->values,
          builder->record.value_count * sizeof(builder->values[0])),
  };

  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_DEVICE_METRIC_SAMPLES;
  metadata.name = stream_name;
  metadata.session_id = session_id;
  metadata.stream_id = record.source_id;
  return iree_hal_profile_sink_write(
      sink, &metadata, builder->record.value_count ? 2 : 1, iovecs);
}

//===----------------------------------------------------------------------===//
// Session lifecycle
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX)

static iree_status_t iree_hal_amdgpu_profile_device_metrics_initialize_source(
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_hal_amdgpu_profile_device_metric_source_t* out_source) {
  memset(out_source, 0, sizeof(*out_source));
  out_source->source_id = (uint64_t)physical_device->device_ordinal + 1u;
  out_source->next_sample_id = 1;
  out_source->physical_device_ordinal =
      (uint32_t)physical_device->device_ordinal;
  out_source->gpu_metrics_file_descriptor = -1;
  for (iree_host_size_t i = 0;
       i < IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_COUNT; ++i) {
    out_source->scalar_file_descriptors[i] = -1;
  }

  if (!physical_device->has_pci_identity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU device metrics require PCI identity for "
                            "physical device %" PRIhsz,
                            physical_device->device_ordinal);
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      out_source->device_path, sizeof(out_source->device_path),
      "/sys/bus/pci/devices/%04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
      physical_device->pci_domain, physical_device->pci_bus,
      physical_device->pci_device, physical_device->pci_function));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      out_source->name, sizeof(out_source->name),
      "amdgpu.sysfs.%04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
      physical_device->pci_domain, physical_device->pci_bus,
      physical_device->pci_device, physical_device->pci_function));

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_device_file(
      out_source, "gpu_metrics", &out_source->gpu_metrics_file_descriptor));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_probe_gpu_metrics(out_source));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_device_file(
      out_source, "gpu_busy_percent",
      iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
          out_source,
          IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_GPU_BUSY_PERCENT)));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_device_file(
      out_source, "mem_busy_percent",
      iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
          out_source,
          IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_BUSY_PERCENT)));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_device_file(
      out_source, "mem_info_vram_used",
      iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
          out_source,
          IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_LOCAL_USED)));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_device_file(
      out_source, "mem_info_vram_total",
      iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
          out_source,
          IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_MEMORY_LOCAL_TOTAL)));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_discover_hwmon(out_source));

  if (out_source->readable_file_count == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "AMDGPU device metrics found no readable sysfs "
                            "files under %s",
                            out_source->device_path);
  }
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_LINUX

iree_status_t iree_hal_amdgpu_profile_device_metrics_session_allocate(
    iree_hal_amdgpu_logical_device_t* logical_device,
    const iree_hal_device_profiling_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_device_metrics_session_t** out_session) {
  *out_session = NULL;
  if (!iree_hal_device_profiling_options_requests_device_metrics(options)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!options->sink)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU device metrics profiling requires a profile sink");
  }
  if (IREE_UNLIKELY(logical_device->physical_device_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU physical device count exceeds uint32_t");
  }

#if defined(IREE_PLATFORM_LINUX)
  iree_host_size_t session_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amdgpu_profile_device_metrics_session_t), &session_size,
      IREE_STRUCT_FIELD(logical_device->physical_device_count,
                        iree_hal_amdgpu_profile_device_metric_source_t, NULL)));

  iree_hal_amdgpu_profile_device_metrics_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, session_size, (void**)&session));
  memset(session, 0, session_size);
  session->host_allocator = host_allocator;
  session->source_count = logical_device->physical_device_count;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < logical_device->physical_device_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_amdgpu_profile_device_metrics_initialize_source(
        logical_device->physical_devices[i], &session->sources[i]);
  }
  if (iree_status_is_ok(status)) {
    *out_session = session;
  } else {
    iree_hal_amdgpu_profile_device_metrics_session_free(session);
  }
  return status;
#else
  (void)host_allocator;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU device metrics require Linux sysfs support");
#endif  // IREE_PLATFORM_LINUX
}

void iree_hal_amdgpu_profile_device_metrics_session_free(
    iree_hal_amdgpu_profile_device_metrics_session_t* session) {
  if (!session) return;
  for (iree_host_size_t i = 0; i < session->source_count; ++i) {
    iree_hal_amdgpu_profile_device_metric_source_t* source =
        &session->sources[i];
    iree_hal_amdgpu_profile_device_metrics_close_file(
        &source->gpu_metrics_file_descriptor);
    for (iree_host_size_t j = 0;
         j < IREE_HAL_AMDGPU_PROFILE_SCALAR_METRIC_SLOT_COUNT; ++j) {
      iree_hal_amdgpu_profile_device_metrics_close_file(
          &source->scalar_file_descriptors[j]);
    }
  }
  iree_allocator_free(session->host_allocator, session);
}

iree_status_t iree_hal_amdgpu_profile_device_metrics_session_write_metadata(
    const iree_hal_amdgpu_profile_device_metrics_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  if (!session) return iree_ok_status();
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < session->source_count && iree_status_is_ok(status); ++i) {
    const iree_hal_amdgpu_profile_device_metric_source_t* source =
        &session->sources[i];
    status = iree_hal_amdgpu_profile_device_metrics_write_source_metadata(
        source, sink, session_id, stream_name);
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(
                 iree_hal_amdgpu_profile_device_metrics_metric_ids) &&
         iree_status_is_ok(status);
         ++j) {
      status = iree_hal_amdgpu_profile_device_metrics_write_descriptor_metadata(
          source, sink, session_id, stream_name,
          iree_hal_amdgpu_profile_device_metrics_metric_ids[j]);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_profile_device_metrics_session_sample_and_write(
    iree_hal_amdgpu_profile_device_metrics_session_t* session,
    iree_hal_profile_sink_t* sink, uint64_t session_id,
    iree_string_view_t stream_name) {
  if (!session) return iree_ok_status();
#if defined(IREE_PLATFORM_LINUX)
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < session->source_count && iree_status_is_ok(status); ++i) {
    iree_hal_amdgpu_profile_device_metric_source_t* source =
        &session->sources[i];
    iree_hal_amdgpu_profile_device_metric_sample_builder_t builder;
    memset(&builder, 0, sizeof(builder));
    builder.record = iree_hal_profile_device_metric_sample_record_default();
    builder.record.sample_id = source->next_sample_id++;
    builder.record.source_id = source->source_id;
    builder.record.physical_device_ordinal = source->physical_device_ordinal;
    builder.record.host_time_begin_ns = iree_time_now();
    status = iree_hal_amdgpu_profile_device_metrics_sample_gpu_metrics(
        source, &builder);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_profile_device_metrics_sample_scalars(source,
                                                                     &builder);
    }
    builder.record.host_time_end_ns = iree_time_now();
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_profile_device_metrics_write_sample(
          &builder, sink, session_id, stream_name);
    }
  }
  return status;
#else
  return iree_ok_status();
#endif  // IREE_PLATFORM_LINUX
}
