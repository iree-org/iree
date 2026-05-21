// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <string.h>

#include "iree/hal/drivers/amdgpu/profile_device_metrics_source.h"

#if defined(IREE_PLATFORM_LINUX)

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"

//===----------------------------------------------------------------------===//
// Linux sysfs/gpu_metrics schema subset
//===----------------------------------------------------------------------===//

#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_GPU_BUFFER_LENGTH 4096
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_MAX_PATH_LENGTH 256
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SCALAR_FILE_CAPACITY \
  10
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_TEXT_BUFFER_LENGTH 64
#define IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_SOURCE_KIND_LINUX_SYSFS 1u

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

// Linux sysfs scalar metric slots discovered once at profiling begin.
typedef uint32_t iree_hal_amdgpu_profile_device_metrics_linux_sysfs_slot_t;
enum iree_hal_amdgpu_profile_device_metrics_linux_sysfs_slot_e {
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_GPU_BUSY_PERCENT = 0,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_BUSY_PERCENT =
      1,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_LOCAL_USED = 2,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_LOCAL_TOTAL =
      3,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_CLOCK_COMPUTE_CURRENT =
      4,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_CLOCK_MEMORY_CURRENT =
      5,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_EDGE = 6,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_HOTSPOT =
      7,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_MEMORY =
      8,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_POWER_SOCKET = 9,
  IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_COUNT =
      IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SCALAR_FILE_CAPACITY,
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

typedef struct iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t {
  // Built-in metric id emitted for this scalar file.
  uint64_t metric_id;

  // Multiplier applied to the parsed unsigned sysfs value.
  uint64_t scale;
} iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t;

static iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t
iree_hal_amdgpu_profile_device_metrics_linux_sysfs_scalar_metric(
    iree_hal_amdgpu_profile_device_metrics_linux_sysfs_slot_t slot) {
  switch (slot) {
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_GPU_BUSY_PERCENT:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_COMPUTE,
          .scale = 1000u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_BUSY_PERCENT:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_ACTIVITY_MEMORY,
          .scale = 1000u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_LOCAL_USED:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_MEMORY_LOCAL_USED,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_LOCAL_TOTAL:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_MEMORY_LOCAL_TOTAL,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_CLOCK_COMPUTE_CURRENT:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_COMPUTE_CURRENT,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_CLOCK_MEMORY_CURRENT:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_CLOCK_MEMORY_CURRENT,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_EDGE:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_EDGE,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_HOTSPOT:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_HOTSPOT,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_MEMORY:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_TEMPERATURE_MEMORY,
          .scale = 1u,
      };
    case IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_POWER_SOCKET:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){
          .metric_id = IREE_HAL_PROFILE_BUILTIN_METRIC_ID_POWER_SOCKET,
          .scale = 1u,
      };
    default:
      return (iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t){0};
  }
}

typedef struct iree_hal_amdgpu_profile_linux_sysfs_source_state_t {
  // Discovered sysfs source identity.
  struct {
    // NUL-terminated sysfs PCI device path.
    char device_path
        [IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_MAX_PATH_LENGTH];

    // Number of readable sysfs files discovered for this source.
    uint32_t readable_file_count;
  } discovery;

  // Open sysfs file descriptors.
  struct {
    // Open gpu_metrics binary sysfs file descriptor, or -1.
    int gpu_metrics;

    // Open scalar sysfs file descriptors indexed by sysfs metric slot, or -1.
    int scalars
        [IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SCALAR_FILE_CAPACITY];
  } files;
} iree_hal_amdgpu_profile_linux_sysfs_source_state_t;

//===----------------------------------------------------------------------===//
// Linux sysfs discovery and sampling
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_profile_device_metrics_linux_sysfs_reset(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state) {
  memset(state, 0, sizeof(*state));
  state->files.gpu_metrics = -1;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(state->files.scalars); ++i) {
    state->files.scalars[i] = -1;
  }
}

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

static void iree_hal_amdgpu_profile_device_metrics_append_u16_scaled(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, uint16_t value, uint64_t scale) {
  if (iree_hal_amdgpu_profile_device_metrics_u16_is_valid(value)) {
    iree_hal_amdgpu_profile_device_metric_sample_builder_append_u64(
        builder, metric_id, (uint64_t)value * scale);
  }
}

static void iree_hal_amdgpu_profile_device_metrics_append_i16_scaled(
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder,
    uint64_t metric_id, uint16_t value, int64_t scale) {
  if (iree_hal_amdgpu_profile_device_metrics_u16_is_valid(value)) {
    iree_hal_amdgpu_profile_device_metric_sample_builder_append_i64(
        builder, metric_id, (int64_t)value * scale);
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

static int* iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
    iree_hal_amdgpu_profile_device_metrics_linux_sysfs_slot_t slot) {
  return &state->files.scalars[slot];
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_open_device_file(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
    const char* file_name, int* out_file_descriptor) {
  char
      path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_MAX_PATH_LENGTH] =
          {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      path, sizeof(path), "%s/%s", state->discovery.device_path, file_name));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_open_optional_file(
          path, out_file_descriptor));
  if (*out_file_descriptor >= 0) ++state->discovery.readable_file_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_read_label(
    const char* directory_path, const char* prefix, uint32_t index,
    char* buffer, iree_host_size_t buffer_capacity, bool* out_available,
    iree_string_view_t* out_label) {
  *out_label = iree_string_view_empty();
  char
      path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_MAX_PATH_LENGTH] =
          {0};
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
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
    iree_hal_amdgpu_profile_device_metrics_linux_sysfs_slot_t slot,
    const char* directory_path, const char* prefix, uint32_t index,
    const char* suffix) {
  if (state->files.scalars[slot] >= 0) {
    return iree_ok_status();
  }
  char
      path[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_MAX_PATH_LENGTH] =
          {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      path, sizeof(path), "%s/%s%" PRIu32 "_%s", directory_path, prefix, index,
      suffix));
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_open_optional_file(
          path, &state->files.scalars[slot]));
  if (state->files.scalars[slot] >= 0) {
    ++state->discovery.readable_file_count;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_discover_hwmon_freq(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
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
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
          state,
          IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_CLOCK_COMPUTE_CURRENT,
          directory_path, "freq", i, "input"));
    } else if (iree_string_view_equal(label, IREE_SV("mclk")) ||
               iree_string_view_equal(label, IREE_SV("uclk"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
          state,
          IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_CLOCK_MEMORY_CURRENT,
          directory_path, "freq", i, "input"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_discover_hwmon_temperature(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
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
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
          state,
          IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_EDGE,
          directory_path, "temp", i, "input"));
    } else if (iree_string_view_equal(label, IREE_SV("junction")) ||
               iree_string_view_equal(label, IREE_SV("hotspot"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
          state,
          IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_HOTSPOT,
          directory_path, "temp", i, "input"));
    } else if (iree_string_view_equal(label, IREE_SV("mem")) ||
               iree_string_view_equal(label, IREE_SV("memory"))) {
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
          state,
          IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_TEMPERATURE_MEMORY,
          directory_path, "temp", i, "input"));
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_profile_device_metrics_discover_hwmon_power(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
    const char* directory_path) {
  return iree_hal_amdgpu_profile_device_metrics_open_hwmon_input(
      state,
      IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_POWER_SOCKET,
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
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state) {
  char directory_path
      [IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_MAX_PATH_LENGTH] = {
          0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_format_path(
      directory_path, sizeof(directory_path), "%s/hwmon",
      state->discovery.device_path));

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
    char hwmon_path
        [IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_MAX_PATH_LENGTH] = {
            0};
    status = iree_hal_amdgpu_profile_device_metrics_format_path(
        hwmon_path, sizeof(hwmon_path), "%s/%s", directory_path, entry->d_name);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_profile_device_metrics_discover_hwmon_freq(
          state, hwmon_path);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_amdgpu_profile_device_metrics_discover_hwmon_temperature(
              state, hwmon_path);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_profile_device_metrics_discover_hwmon_power(
          state, hwmon_path);
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
    iree_hal_amdgpu_profile_device_metric_sample_builder_append_u64(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_THROTTLE_STATUS,
        metrics->indep_throttle_status);
  } else if (IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD(
                 storage_length, iree_hal_amdgpu_profile_gpu_metrics_v1_3_t,
                 throttle_status) &&
             iree_hal_amdgpu_profile_device_metrics_u32_is_valid(
                 metrics->throttle_status)) {
    iree_hal_amdgpu_profile_device_metric_sample_builder_append_u64(
        builder, IREE_HAL_PROFILE_BUILTIN_METRIC_ID_THROTTLE_STATUS,
        metrics->throttle_status);
  }
}

#undef IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_HAS_FIELD

static iree_status_t iree_hal_amdgpu_profile_device_metrics_parse_gpu_metrics(
    const uint8_t* storage, iree_host_size_t storage_length,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  if (storage_length < sizeof(iree_hal_amdgpu_profile_metrics_table_header_t)) {
    builder->record.flags |= IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
    return iree_ok_status();
  }
  const iree_hal_amdgpu_profile_metrics_table_header_t* header =
      (const iree_hal_amdgpu_profile_metrics_table_header_t*)(const void*)
          storage;
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
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
    uint32_t* out_source_revision) {
  iree_alignas(8)
      uint8_t storage[sizeof(iree_hal_amdgpu_profile_metrics_table_header_t)] =
          {0};
  bool available = false;
  iree_host_size_t storage_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_read_file(
      state->files.gpu_metrics, "gpu_metrics", storage, sizeof(storage),
      &available, &storage_length));
  if (!available ||
      storage_length < sizeof(iree_hal_amdgpu_profile_metrics_table_header_t)) {
    return iree_ok_status();
  }

  const iree_hal_amdgpu_profile_metrics_table_header_t* header =
      (const iree_hal_amdgpu_profile_metrics_table_header_t*)(const void*)
          storage;
  *out_source_revision =
      ((uint32_t)header->format_revision << 8) | header->content_revision;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_sample_gpu_metrics(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  iree_alignas(8) uint8_t
      storage[IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_GPU_BUFFER_LENGTH] = {0};
  bool available = false;
  iree_host_size_t storage_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_device_metrics_read_file(
      state->files.gpu_metrics, "gpu_metrics", storage, sizeof(storage),
      &available, &storage_length));
  if (!available) {
    if (state->files.gpu_metrics >= 0) {
      builder->record.flags |=
          IREE_HAL_PROFILE_DEVICE_METRIC_SAMPLE_FLAG_PARTIAL;
    }
    return iree_ok_status();
  }
  return iree_hal_amdgpu_profile_device_metrics_parse_gpu_metrics(
      storage, storage_length, builder);
}

static iree_status_t iree_hal_amdgpu_profile_device_metrics_sample_scalars(
    iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  for (iree_host_size_t i = 0;
       i < IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_COUNT; ++i) {
    const int file_descriptor = state->files.scalars[i];
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
    const iree_hal_amdgpu_profile_linux_sysfs_scalar_metric_t scalar_metric =
        iree_hal_amdgpu_profile_device_metrics_linux_sysfs_scalar_metric(
            (iree_hal_amdgpu_profile_device_metrics_linux_sysfs_slot_t)i);
    iree_hal_amdgpu_profile_device_metric_sample_builder_append_u64(
        builder, scalar_metric.metric_id, value * scalar_metric.scale);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_profile_device_metric_source_initialize(
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_device_metric_source_t* out_source) {
  memset(out_source, 0, sizeof(*out_source));
  if (IREE_UNLIKELY(physical_device->device_ordinal > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU device metric source physical device ordinal exceeds uint32_t");
  }
  if (!physical_device->has_pci_identity) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU device metrics require PCI identity for "
                            "physical device %" PRIhsz,
                            physical_device->device_ordinal);
  }

  iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*state), (void**)&state));
  iree_hal_amdgpu_profile_device_metrics_linux_sysfs_reset(state);
  out_source->platform.host_allocator = host_allocator;
  out_source->platform.state = state;
  out_source->metadata.id = (uint64_t)physical_device->device_ordinal + 1u;
  out_source->metadata.kind =
      IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_SOURCE_KIND_LINUX_SYSFS;
  out_source->metadata.physical_device_ordinal =
      (uint32_t)physical_device->device_ordinal;
  out_source->metrics.count =
      IREE_ARRAYSIZE(iree_hal_amdgpu_profile_device_metrics_metric_ids);
  out_source->metrics.ids = iree_hal_amdgpu_profile_device_metrics_metric_ids;
  out_source->sampling.next_sample_id = 1;

  iree_status_t status = iree_hal_amdgpu_profile_device_metrics_format_path(
      state->discovery.device_path, sizeof(state->discovery.device_path),
      "/sys/bus/pci/devices/%04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
      physical_device->pci_domain, physical_device->pci_bus,
      physical_device->pci_device, physical_device->pci_function);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_format_path(
        out_source->metadata.name, sizeof(out_source->metadata.name),
        "amdgpu.sysfs.%04" PRIx32 ":%02" PRIx32 ":%02" PRIx32 ".%" PRIu32,
        physical_device->pci_domain, physical_device->pci_bus,
        physical_device->pci_device, physical_device->pci_function);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_open_device_file(
        state, "gpu_metrics", &state->files.gpu_metrics);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_probe_gpu_metrics(
        state, &out_source->metadata.revision);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_open_device_file(
        state, "gpu_busy_percent",
        iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
            state,
            IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_GPU_BUSY_PERCENT));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_open_device_file(
        state, "mem_busy_percent",
        iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
            state,
            IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_BUSY_PERCENT));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_open_device_file(
        state, "mem_info_vram_used",
        iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
            state,
            IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_LOCAL_USED));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_open_device_file(
        state, "mem_info_vram_total",
        iree_hal_amdgpu_profile_device_metrics_scalar_file_descriptor(
            state,
            IREE_HAL_AMDGPU_PROFILE_DEVICE_METRICS_LINUX_SYSFS_SLOT_MEMORY_LOCAL_TOTAL));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_profile_device_metrics_discover_hwmon(state);
  }
  if (iree_status_is_ok(status) && state->discovery.readable_file_count == 0) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "AMDGPU device metrics found no readable sysfs "
                              "files under %s",
                              state->discovery.device_path);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_profile_device_metric_source_deinitialize(out_source);
  }
  return status;
}

void iree_hal_amdgpu_profile_device_metric_source_deinitialize(
    iree_hal_amdgpu_profile_device_metric_source_t* source) {
  if (!source) return;
  iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state =
      (iree_hal_amdgpu_profile_linux_sysfs_source_state_t*)
          source->platform.state;
  if (!state) {
    memset(source, 0, sizeof(*source));
    return;
  }
  iree_hal_amdgpu_profile_device_metrics_close_file(&state->files.gpu_metrics);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(state->files.scalars); ++i) {
    iree_hal_amdgpu_profile_device_metrics_close_file(&state->files.scalars[i]);
  }
  iree_allocator_free(source->platform.host_allocator, state);
  memset(source, 0, sizeof(*source));
}

iree_status_t iree_hal_amdgpu_profile_device_metric_source_sample(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  iree_hal_amdgpu_profile_linux_sysfs_source_state_t* state =
      (iree_hal_amdgpu_profile_linux_sysfs_source_state_t*)
          source->platform.state;
  if (IREE_UNLIKELY(!state)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU device metric source is uninitialized");
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_profile_device_metrics_sample_gpu_metrics(state,
                                                                builder));
  return iree_hal_amdgpu_profile_device_metrics_sample_scalars(state, builder);
}

#else

iree_status_t iree_hal_amdgpu_profile_device_metric_source_initialize(
    iree_hal_amdgpu_physical_device_t* physical_device,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_device_metric_source_t* out_source) {
  (void)physical_device;
  (void)host_allocator;
  memset(out_source, 0, sizeof(*out_source));
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU device metrics require Linux sysfs support");
}

void iree_hal_amdgpu_profile_device_metric_source_deinitialize(
    iree_hal_amdgpu_profile_device_metric_source_t* source) {
  if (!source) return;
  memset(source, 0, sizeof(*source));
}

iree_status_t iree_hal_amdgpu_profile_device_metric_source_sample(
    iree_hal_amdgpu_profile_device_metric_source_t* source,
    iree_hal_amdgpu_profile_device_metric_sample_builder_t* builder) {
  (void)source;
  (void)builder;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "AMDGPU device metrics require Linux sysfs support");
}

#endif  // IREE_PLATFORM_LINUX
