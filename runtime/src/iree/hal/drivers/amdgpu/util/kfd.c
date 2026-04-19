// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/kfd.h"

#include <inttypes.h>
#include <string.h>  // memset

//===----------------------------------------------------------------------===//
// KFD IOCTL Workaround
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX)

#include <errno.h>
#include <fcntl.h>      // open
#include <sys/ioctl.h>  // ioctl
#include <unistd.h>     // close

#define IREE_AMDKFD_IOCTL_BASE 'K'
#define IREE_AMDKFD_IO(nr) _IO(IREE_AMDKFD_IOCTL_BASE, nr)
#define IREE_AMDKFD_IOR(nr, type) _IOR(IREE_AMDKFD_IOCTL_BASE, nr, type)
#define IREE_AMDKFD_IOW(nr, type) _IOW(IREE_AMDKFD_IOCTL_BASE, nr, type)
#define IREE_AMDKFD_IOWR(nr, type) _IOWR(IREE_AMDKFD_IOCTL_BASE, nr, type)

iree_status_t iree_hal_amdgpu_kfd_open(int* out_fd) {
  IREE_ASSERT_ARGUMENT(out_fd);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_fd = -1;

  iree_status_t status = iree_ok_status();
  const int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (fd == -1) {
    const int errsv = errno;
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "unable to open /dev/kfd channel; platform file "
                              "handle limit may be reached; errno=%d (%s)",
                              errsv, strerror(errsv));
  }

  if (iree_status_is_ok(status)) {
    *out_fd = fd;
  } else if (fd >= 0) {
    close(fd);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_kfd_close(int fd) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (fd >= 0) {
    close(fd);
  }
  IREE_TRACE_ZONE_END(z0);
}

int iree_hal_amdgpu_ioctl(int fd, unsigned long request, void* arg) {
  int ret = 0;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
  return ret;
}

#else

iree_status_t iree_hal_amdgpu_kfd_open(int* out_fd) {
  IREE_ASSERT_ARGUMENT(out_fd);
  *out_fd = -1;
  return iree_ok_status();
}

void iree_hal_amdgpu_kfd_close(int fd) { (void)fd; }

int iree_hal_amdgpu_ioctl(int fd, unsigned long request, void* arg) {
  (void)fd;
  (void)request;
  (void)arg;
  return -1;
}

#endif  // IREE_PLATFORM_LINUX

//===----------------------------------------------------------------------===//
// AMDKFD_IOC_GET_CLOCK_COUNTERS
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_amdgpu_kfd_validate_clock_counters(
    uint32_t gpu_uid, const iree_hal_amdgpu_clock_counters_t* counters) {
  IREE_ASSERT_ARGUMENT(counters);
  if (IREE_UNLIKELY(counters->gpu_clock_counter == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "KFD returned an invalid zero gpu_clock_counter for gpu_uid=%" PRIu32,
        gpu_uid);
  }
  if (IREE_UNLIKELY(counters->cpu_clock_counter == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "KFD returned an invalid zero cpu_clock_counter for gpu_uid=%" PRIu32,
        gpu_uid);
  }
  if (IREE_UNLIKELY(counters->system_clock_counter == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "KFD returned an invalid zero "
                            "system_clock_counter for gpu_uid=%" PRIu32,
                            gpu_uid);
  }
  if (IREE_UNLIKELY(counters->system_clock_freq == 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "KFD returned an invalid zero system_clock_freq for gpu_uid=%" PRIu32,
        gpu_uid);
  }
  return iree_ok_status();
}

#if defined(IREE_PLATFORM_LINUX)

#define IREE_AMDKFD_IOC_GET_CLOCK_COUNTERS \
  IREE_AMDKFD_IOWR(0x05, struct iree_kfd_ioctl_get_clock_counters_args)

struct iree_kfd_ioctl_get_clock_counters_args {
  // GPU clock counter returned by KFD.
  uint64_t gpu_clock_counter;

  // Host CPU timestamp returned by KFD.
  uint64_t cpu_clock_counter;

  // Host system clock counter returned by KFD.
  uint64_t system_clock_counter;

  // Frequency in Hz for system_clock_counter returned by KFD.
  uint64_t system_clock_freq;

  // GPU identifier passed to KFD.
  uint32_t gpu_id;

  // Reserved padding matching the KFD ABI.
  uint32_t pad;
};

iree_status_t iree_hal_amdgpu_kfd_get_clock_counters(
    int fd, uint32_t gpu_uid, iree_hal_amdgpu_clock_counters_t* out_counters) {
  IREE_ASSERT_ARGUMENT(out_counters);
  memset(out_counters, 0, sizeof(*out_counters));
  if (IREE_UNLIKELY(fd < 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid /dev/kfd file descriptor for "
                            "AMDKFD_IOC_GET_CLOCK_COUNTERS(gpu_uid=%" PRIu32
                            ")",
                            gpu_uid);
  }

  struct iree_kfd_ioctl_get_clock_counters_args args = {0};
  args.gpu_id = gpu_uid;
  int kmt_err =
      iree_hal_amdgpu_ioctl(fd, IREE_AMDKFD_IOC_GET_CLOCK_COUNTERS, &args);
  if (IREE_UNLIKELY(kmt_err < 0)) {
    const int errsv = errno;
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDKFD_IOC_GET_CLOCK_COUNTERS(gpu_uid=%" PRIu32
                            ") failed with %d; errno=%d (%s)",
                            gpu_uid, kmt_err, errsv, strerror(errsv));
  }
  out_counters->gpu_clock_counter = args.gpu_clock_counter;
  out_counters->cpu_clock_counter = args.cpu_clock_counter;
  out_counters->system_clock_counter = args.system_clock_counter;
  out_counters->system_clock_freq = args.system_clock_freq;
  return iree_hal_amdgpu_kfd_validate_clock_counters(gpu_uid, out_counters);
}

#else

iree_status_t iree_hal_amdgpu_kfd_get_clock_counters(
    int fd, uint32_t gpu_uid, iree_hal_amdgpu_clock_counters_t* out_counters) {
  (void)fd;
  (void)gpu_uid;
  memset(out_counters, 0, sizeof(*out_counters));
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "AMDKFD_IOC_GET_CLOCK_COUNTERS requires Linux /dev/kfd support");
}

#endif  // IREE_PLATFORM_LINUX
