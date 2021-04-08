// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Only compile if an external implementation has not been otherwise linked.
#if !defined(VULKAN_MEMORY_ALLOCATOR_EXTERNAL_IMPL)

#include "iree/base/logging.h"
#include "iree/base/synchronization.h"

// Use std::unordered_map.
#define VmaPair std::pair
#define VMA_MAP_TYPE(KeyT, ValueT)                                       \
  std::unordered_map<KeyT, ValueT, std::hash<KeyT>, std::equal_to<KeyT>, \
                     VmaStlAllocator<std::pair<KeyT, ValueT> > >

// Use IREE_CHECK for assertions.
#define VMA_ASSERT IREE_CHECK
#define VMA_HEAVY_ASSERT IREE_DCHECK

// Use IREE_LOG for logging.
#ifndef NDEBUG
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
#include "iree/hal/vulkan/internal_vk_mem_alloc.h"

#endif  // !VULKAN_MEMORY_ALLOCATOR_EXTERNAL_IMPL
