// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Only compile if an external implementation has not been otherwise linked.
#if !defined(VULKAN_MEMORY_ALLOCATOR_EXTERNAL_IMPL)

#include <ostream>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"

#define VMA_ASSERT IREE_ASSERT
#define VMA_HEAVY_ASSERT IREE_ASSERT

// NOTE: logging is disabled by default as unless you are debugging VMA itself
// the information is not useful and just slows things down.
#if 0
#define VMA_DEBUG_LOG(...) _IREE_LOG_INFO << __VA_ARGS__
#else
#define VMA_DEBUG_LOG(...)
#endif  // !NDEBUG

// Use iree_slim_mutex_t for VMA_MUTEX.
class IreeVmaMutex {
 public:
  IreeVmaMutex() { iree_slim_mutex_initialize(&mutex_); }
  ~IreeVmaMutex() { iree_slim_mutex_deinitialize(&mutex_); }

  void Lock() { iree_slim_mutex_lock(&mutex_); }
  void Unlock() { iree_slim_mutex_unlock(&mutex_); }
  bool TryLock() { return iree_slim_mutex_try_lock(&mutex_); }

 private:
  iree_slim_mutex_t mutex_;
};
#define VMA_MUTEX IreeVmaMutex

// Use iree_slim_mutex_t for VMA_RW_MUTEX.
class IreeVmaRWMutex {
 public:
  IreeVmaRWMutex() { iree_slim_mutex_initialize(&mutex_); }
  ~IreeVmaRWMutex() { iree_slim_mutex_deinitialize(&mutex_); }

  void LockRead() { iree_slim_mutex_lock(&mutex_); }
  void UnlockRead() { iree_slim_mutex_unlock(&mutex_); }
  bool TryLockRead() { return iree_slim_mutex_try_lock(&mutex_); }
  void LockWrite() { iree_slim_mutex_lock(&mutex_); }
  void UnlockWrite() { iree_slim_mutex_unlock(&mutex_); }
  bool TryLockWrite() { return iree_slim_mutex_try_lock(&mutex_); }

 private:
  iree_slim_mutex_t mutex_;
};
#define VMA_RW_MUTEX IreeVmaRWMutex

#define VMA_IMPLEMENTATION
#include "iree/hal/drivers/vulkan/internal_vk_mem_alloc.h"

#endif  // !VULKAN_MEMORY_ALLOCATOR_EXTERNAL_IMPL
