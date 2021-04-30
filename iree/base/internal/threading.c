// Copyright 2020 Google LLC
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

#include "iree/base/internal/threading.h"

#include "iree/base/internal/threading_impl.h"

#if defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
#include <xmmintrin.h>
#endif  // IREE_ARCH_X86_*

#if defined(IREE_COMPILER_MSVC)
#include <intrin.h>
#endif  // IREE_COMPILER_MSVC

int iree_strncpy_s(char* IREE_RESTRICT dest, size_t destsz,
                   const char* IREE_RESTRICT src, size_t count) {
#if defined(IREE_COMPILER_MSVC) || defined(__STDC_LIB_EXT1__)
  return strncpy_s(dest, destsz, src, count);
#else
  if (!src || !dest || !destsz) return EINVAL;
  size_t src_len = strnlen(src, destsz);
  if (count >= destsz && destsz <= src_len) return ERANGE;
  if (src_len > count) src_len = count;
  while (*src != 0 && src_len > 0) {
    *(dest++) = *(src++);
    --src_len;
  }
  *dest = 0;
  return 0;
#endif  // GNU
}

//==============================================================================
// iree_thread_affinity_t
//==============================================================================

// TODO(benvanik): add more helpers and possibly move cpuinfo usage into here.

void iree_thread_affinity_set_any(iree_thread_affinity_t* out_thread_affinity) {
  memset(out_thread_affinity, 0x00, sizeof(*out_thread_affinity));
}

//==============================================================================
// iree_thread_override_list_t
//==============================================================================
// This is shared by multiple platform implementations and gets stripped in LTO
// when unused.

struct iree_thread_override_s {
  iree_thread_override_list_t* list;
  iree_thread_override_t* next;
  iree_thread_override_t* prev;
  iree_thread_t* thread;
  iree_thread_priority_class_t priority_class;
};

void iree_thread_override_list_initialize(
    iree_thread_set_priority_fn_t set_priority_fn,
    iree_thread_priority_class_t base_priority_class,
    iree_allocator_t allocator, iree_thread_override_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  out_list->set_priority_fn = set_priority_fn;
  out_list->base_priority_class = base_priority_class;
  out_list->allocator = allocator;
  iree_slim_mutex_initialize(&out_list->mutex);
  out_list->current_priority_class = base_priority_class;
}

void iree_thread_override_list_deinitialize(iree_thread_override_list_t* list) {
#if !defined(NDEBUG)
  // Assert that all overrides have been removed (and properly freed).
  iree_slim_mutex_lock(&list->mutex);
  assert(!list->head);
  iree_slim_mutex_unlock(&list->mutex);
#endif  // !NDEBUG

  iree_slim_mutex_deinitialize(&list->mutex);
}

// Updates the priority class of the thread to the maximum across all overrides
// and the base thread priority class.
//
// NOTE: assumes the lock is held so the list can be safely walked.
static void iree_thread_override_list_update_priority_class(
    iree_thread_override_list_t* list, iree_thread_t* thread) {
  // Compute the new maximum priority class with the override now added.
  iree_thread_priority_class_t max_priority_class = list->base_priority_class;
  for (iree_thread_override_t* override = list->head; override != NULL;
       override = override->next) {
    max_priority_class = iree_max(max_priority_class, override->priority_class);
  }
  bool needs_update = max_priority_class != list->current_priority_class;
  list->current_priority_class = max_priority_class;

  // Change priority if needed (this way we are avoiding syscalls if we get a
  // wave of overrides at the same priority class).
  //
  // NOTE: we do this inside the lock so that we don't lose priorities. It'd be
  // nice to do this outside the lock if we could so we aren't holding it during
  // a syscall. Overrides should (hopefully) be infrequent enough that this is
  // rarely called.
  if (needs_update) {
    list->set_priority_fn(thread, max_priority_class);
  }
}

iree_thread_override_t* iree_thread_override_list_add(
    iree_thread_override_list_t* list, iree_thread_t* thread,
    iree_thread_priority_class_t priority_class) {
  // Allocate the override struct we'll pass back to the caller.
  iree_thread_override_t* override = NULL;
  iree_status_t status = iree_allocator_malloc(
      list->allocator, sizeof(*override), (void**)&override);
  if (IREE_UNLIKELY(!iree_status_is_ok(iree_status_consume_code(status)))) {
    return NULL;
  }
  override->list = list;
  override->next = NULL;
  override->prev = NULL;
  override->thread = thread;
  override->priority_class = priority_class;

  iree_slim_mutex_lock(&list->mutex);

  // Add the override to the list.
  override->next = list->head;
  if (list->head) {
    list->head->prev = override;
  }
  list->head = override;

  // Update and change priority if needed.
  // NOTE: the lock must be held.
  iree_thread_override_list_update_priority_class(list, thread);

  iree_slim_mutex_unlock(&list->mutex);

  return override;
}

void iree_thread_override_remove_self(iree_thread_override_t* override) {
  iree_thread_override_list_t* list = override->list;
  iree_slim_mutex_lock(&list->mutex);

  // Remove the override from the list.
  if (override->prev) {
    override->prev->next = override->next;
  }
  if (override->next) {
    override->next->prev = override->prev;
  }
  if (list->head == override) {
    list->head = override->next;
  }

  // Update and change priority if needed.
  // NOTE: the lock must be held.
  iree_thread_t* thread = override->thread;
  iree_thread_override_list_update_priority_class(list, thread);

  iree_slim_mutex_unlock(&list->mutex);

  // Deallocate the override outside of the lock as no one should be using it
  // anymore.
  iree_allocator_free(list->allocator, override);
}

//==============================================================================
// iree_fpu_state_t
//==============================================================================
// https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/core/platform/denormal.cc
// https://chromium.googlesource.com/chromium/blink/+/master/Source/platform/audio/DenormalDisabler.h

static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero);

#if defined(IREE_ARCH_ARM_32)
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return (state & ~0x1000000) | (denormals_to_zero ? 0x1000000 : 0);
}
#elif defined(IREE_ARCH_ARM_64)
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return (state & ~0x1080000) | (denormals_to_zero ? 0x1080000 : 0);
}
#elif defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return (state & ~0x8040) | (denormals_to_zero ? 0x8040 : 0);
}
#else
static uint64_t iree_fpu_state_set_dtz(uint64_t state, bool denormals_to_zero) {
  return state;
}
#endif  // IREE_ARCH_*

static uint64_t iree_fpu_load_state();
static void iree_fpu_store_state(uint64_t state);

#if defined(IREE_ARCH_ARM_32) && defined(IREE_COMPILER_MSVC)
static uint64_t iree_fpu_load_state() {
  return (uint64_t)_MoveFromCoprocessor(10, 7, 1, 0, 0);
}
static void iree_fpu_store_state(uint64_t state) {
  _MoveToCoprocessor((int)state, 10, 7, 1, 0, 0);
}
#elif defined(IREE_ARCH_ARM_32)
static uint64_t iree_fpu_load_state() {
  uint32_t fpscr;
  __asm__ __volatile__("VMRS %[fpscr], fpscr" : [ fpscr ] "=r"(fpscr));
  return (uint64_t)fpscr;
}
static void iree_fpu_store_state(uint64_t state) {
  __asm__ __volatile__("VMSR fpscr, %[fpscr]" : : [ fpscr ] "r"(state));
}
#elif defined(IREE_ARCH_ARM_64) && defined(IREE_COMPILER_MSVC)
static uint64_t iree_fpu_load_state() {
  return (uint64_t)_ReadStatusReg(0x5A20);
}
static void iree_fpu_store_state(uint64_t state) {
  _WriteStatusReg(0x5A20, (__int64)state);
}
#elif defined(IREE_ARCH_ARM_64)
static uint64_t iree_fpu_load_state() {
  uint64_t fpcr;
  __asm__ __volatile__("MRS %[fpcr], fpcr" : [ fpcr ] "=r"(fpcr));
  return fpcr;
}
static void iree_fpu_store_state(uint64_t state) {
  __asm__ __volatile__("MSR fpcr, %[fpcr]" : : [ fpcr ] "r"(state));
}
#elif defined(IREE_ARCH_X86_32) || defined(IREE_ARCH_X86_64)
static uint64_t iree_fpu_load_state() { return (uint64_t)_mm_getcsr(); }
static void iree_fpu_store_state(uint64_t state) {
  _mm_setcsr((unsigned int)state);
}
#else
static uint64_t iree_fpu_load_state() { return 0; }
static void iree_fpu_store_state(uint64_t state) {}
#endif  // IREE_ARCH_*

iree_fpu_state_t iree_fpu_state_push(iree_fpu_state_flags_t flags) {
  iree_fpu_state_t state;
  state.current_value = state.previous_value = iree_fpu_load_state();
  state.current_value = iree_fpu_state_set_dtz(
      state.current_value,
      (flags & IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO) ? true : false);
  if (state.previous_value != state.current_value) {
    iree_fpu_store_state(state.current_value);
  }
  return state;
}

void iree_fpu_state_pop(iree_fpu_state_t state) {
  if (state.previous_value != state.current_value) {
    iree_fpu_store_state(state.previous_value);
  }
}
