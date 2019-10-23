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

// This file configures VMA to use common Google/Abseil types in an effort to
// better integrate with applications compiled using other Google code. By using
// the same types that dependers are likely using we can often reduce binary
// size and ease debugging (such as by using absl::Mutex to get better tsan
// warnings).

// Only compile if an external implementation has not been otherwise linked.
#if !defined(VULKAN_MEMORY_ALLOCATOR_EXTERNAL_IMPL)

#include "absl/container/flat_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "iree/base/logging.h"

// Use std::vector instead of the VMA version.
#define VMA_USE_STL_VECTOR 1

// TODO(benvanik): figure out why std::list cannot be used.
// #define VMA_USE_STL_LIST 1

// Use absl::flat_hash_map instead of std::unordered_map.
#define VmaPair std::pair
#define VMA_MAP_TYPE(KeyT, ValueT)                                        \
  absl::flat_hash_map<KeyT, ValueT, std::hash<KeyT>, std::equal_to<KeyT>, \
                      VmaStlAllocator<std::pair<KeyT, ValueT> > >

// Use CHECK for assertions.
#define VMA_ASSERT CHECK
#define VMA_HEAVY_ASSERT DCHECK

// Use LOG for logging.
#ifndef NDEBUG
#define VMA_DEBUG_LOG(...) ABSL_RAW_LOG(INFO, __VA_ARGS__)
#else
#define VMA_DEBUG_LOG(...)
#endif  // !NDEBUG

// Use absl::Mutex for VMA_MUTEX.
#define VMA_MUTEX absl::Mutex
class AbslVmaRWMutex {
 public:
  void LockRead() ABSL_SHARED_LOCK_FUNCTION() { mutex_.ReaderLock(); }
  void UnlockRead() ABSL_UNLOCK_FUNCTION() { mutex_.ReaderUnlock(); }
  void LockWrite() ABSL_EXCLUSIVE_LOCK_FUNCTION() { mutex_.WriterLock(); }
  void UnlockWrite() ABSL_UNLOCK_FUNCTION() { mutex_.WriterUnlock(); }

 private:
  absl::Mutex mutex_;
};
#define VMA_RW_MUTEX AbslVmaRWMutex

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#endif
