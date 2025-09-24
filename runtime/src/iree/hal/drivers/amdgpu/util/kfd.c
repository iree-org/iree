// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/kfd.h"

//===----------------------------------------------------------------------===//
// KFD IOCTL Workaround
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX)

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
  *out_fd = 0;

  iree_status_t status = iree_ok_status();
  const int fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (fd == -1) {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "unable to open /dev/kfd channel; platform file "
                              "handle limit may be reached");
  }

  if (iree_status_is_ok(status)) {
    *out_fd = fd;
  } else if (fd > 0) {
    close(fd);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_kfd_close(int fd) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (fd > 0) {
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
  *out_fd = 0;
  return iree_ok_status();
}

void iree_hal_amdgpu_kfd_close(int fd) {}

int iree_hal_amdgpu_ioctl(int fd, unsigned long request, void* arg) {
  return -1;
}

#endif  // IREE_PLATFORM_LINUX

//===----------------------------------------------------------------------===//
// AMDKFD_IOC_GET_CLOCK_COUNTERS
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_LINUX)

#define IREE_AMDKFD_IOC_GET_CLOCK_COUNTERS \
  IREE_AMDKFD_IOWR(0x05, struct iree_kfd_ioctl_get_clock_counters_args)

struct iree_kfd_ioctl_get_clock_counters_args {
  uint64_t gpu_clock_counter;     // from KFD
  uint64_t cpu_clock_counter;     // from KFD
  uint64_t system_clock_counter;  // from KFD
  uint64_t system_clock_freq;     // from KFD
  uint32_t gpu_id;                // to KFD
  uint32_t pad;
};

iree_status_t iree_hal_amdgpu_kfd_get_clock_counters(
    int fd, uint32_t gpu_uid, iree_hal_amdgpu_clock_counters_t* out_counters) {
  struct iree_kfd_ioctl_get_clock_counters_args args = {0};
  args.gpu_id = gpu_uid;
  int kmt_err =
      iree_hal_amdgpu_ioctl(fd, IREE_AMDKFD_IOC_GET_CLOCK_COUNTERS, &args);
  if (IREE_UNLIKELY(kmt_err < 0)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "AMDKFD_IOC_GET_CLOCK_COUNTERS failed with %d",
                            kmt_err);
  }
  out_counters->gpu_clock_counter = args.gpu_clock_counter;
  out_counters->cpu_clock_counter = args.cpu_clock_counter;
  out_counters->system_clock_counter = args.system_clock_counter;
  out_counters->system_clock_freq = args.system_clock_freq;
  return iree_ok_status();
}

#else

iree_status_t iree_hal_amdgpu_kfd_get_clock_counters(
    int fd, uint32_t gpu_uid, iree_hal_amdgpu_clock_counters_t* out_counters) {
  memset(out_counters, 0, sizeof(*out_counters));
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_LINUX
