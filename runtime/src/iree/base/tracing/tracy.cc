// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/tracing.h"

// Textually include the Tracy implementation.
// We do this here instead of relying on an external build target so that we can
// ensure our configuration specified in tracing.h is picked up.
#if IREE_TRACING_FEATURES != 0
#include "TracyClient.cpp"
#endif  // IREE_TRACING_FEATURES

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#if defined(TRACY_ENABLE) && defined(IREE_PLATFORM_WINDOWS)
static HANDLE iree_dbghelp_mutex;
void IREEDbgHelpInit(void) {
  iree_dbghelp_mutex = CreateMutex(NULL, FALSE, NULL);
}
void IREEDbgHelpLock(void) {
  WaitForSingleObject(iree_dbghelp_mutex, INFINITE);
}
void IREEDbgHelpUnlock(void) { ReleaseMutex(iree_dbghelp_mutex); }
#endif  // TRACY_ENABLE && IREE_PLATFORM_WINDOWS

#if IREE_TRACING_FEATURES != 0

void iree_tracing_tracy_initialize() {
  // No-op.
}

void iree_tracing_tracy_deinitialize() {
#if defined(IREE_PLATFORM_APPLE)
  // Synchronously shut down the profiler service.
  // This is required on some platforms to support TRACY_NO_EXIT=1 such as
  // MacOS and iOS. It should be harmless on other platforms as it returns
  // quickly if TRACY_NO_EXIT=1 is not set.
  // See: https://github.com/wolfpld/tracy/issues/8
  tracy::GetProfiler().RequestShutdown();
  while (!tracy::GetProfiler().HasShutdownFinished()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
#endif  // IREE_PLATFORM_*
}

iree_zone_id_t iree_tracing_zone_begin_impl(
    const iree_tracing_location_t* src_loc, const char* name,
    size_t name_length) {
  const iree_zone_id_t zone_id = tracy::GetProfiler().GetNextZoneId();

#ifndef TRACY_NO_VERIFY
  {
    TracyQueuePrepareC(tracy::QueueType::ZoneValidation);
    tracy::MemWrite(&item->zoneValidation.id, zone_id);
    TracyQueueCommitC(zoneValidationThread);
  }
#endif  // TRACY_NO_VERIFY

  {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    TracyQueuePrepareC(tracy::QueueType::ZoneBeginCallstack);
#else
    TracyQueuePrepareC(tracy::QueueType::ZoneBegin);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    tracy::MemWrite(&item->zoneBegin.time, tracy::Profiler::GetTime());
    tracy::MemWrite(&item->zoneBegin.srcloc,
                    reinterpret_cast<uint64_t>(src_loc));
    TracyQueueCommitC(zoneBeginThread);
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
  tracy::GetProfiler().SendCallstack(IREE_TRACING_MAX_CALLSTACK_DEPTH);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS

  if (name_length) {
#ifndef TRACY_NO_VERIFY
    {
      TracyQueuePrepareC(tracy::QueueType::ZoneValidation);
      tracy::MemWrite(&item->zoneValidation.id, zone_id);
      TracyQueueCommitC(zoneValidationThread);
    }
#endif  // TRACY_NO_VERIFY
    auto name_ptr = reinterpret_cast<char*>(tracy::tracy_malloc(name_length));
    memcpy(name_ptr, name, name_length);
    TracyQueuePrepareC(tracy::QueueType::ZoneName);
    tracy::MemWrite(&item->zoneTextFat.text,
                    reinterpret_cast<uint64_t>(name_ptr));
    tracy::MemWrite(&item->zoneTextFat.size,
                    static_cast<uint64_t>(name_length));
    TracyQueueCommitC(zoneTextFatThread);
  }

  return zone_id;
}

iree_zone_id_t iree_tracing_zone_begin_external_impl(
    const char* file_name, size_t file_name_length, uint32_t line,
    const char* function_name, size_t function_name_length, const char* name,
    size_t name_length) {
  uint64_t src_loc = tracy::Profiler::AllocSourceLocation(
      line, file_name, file_name_length, function_name, function_name_length,
      name, name_length);

  const iree_zone_id_t zone_id = tracy::GetProfiler().GetNextZoneId();

#ifndef TRACY_NO_VERIFY
  {
    TracyQueuePrepareC(tracy::QueueType::ZoneValidation);
    tracy::MemWrite(&item->zoneValidation.id, zone_id);
    TracyQueueCommitC(zoneValidationThread);
  }
#endif  // TRACY_NO_VERIFY

  {
#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    TracyQueuePrepareC(tracy::QueueType::ZoneBeginAllocSrcLocCallstack);
#else
    TracyQueuePrepareC(tracy::QueueType::ZoneBeginAllocSrcLoc);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
    tracy::MemWrite(&item->zoneBegin.time, tracy::Profiler::GetTime());
    tracy::MemWrite(&item->zoneBegin.srcloc, src_loc);
    TracyQueueCommitC(zoneBeginThread);
  }

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS
  tracy::GetProfiler().SendCallstack(IREE_TRACING_MAX_CALLSTACK_DEPTH);
#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_CALLSTACKS

  return zone_id;
}

void iree_tracing_zone_end(iree_zone_id_t zone_id) {
  ___tracy_emit_zone_end(iree_tracing_make_zone_ctx(zone_id));
}

void iree_tracing_set_plot_type_impl(const char* name_literal,
                                     uint8_t plot_type, bool step, bool fill,
                                     uint32_t color) {
  tracy::Profiler::ConfigurePlot(name_literal,
                                 static_cast<tracy::PlotFormatType>(plot_type),
                                 step, fill, color);
}

void iree_tracing_plot_value_i64_impl(const char* name_literal, int64_t value) {
  tracy::Profiler::PlotData(name_literal, value);
}

void iree_tracing_plot_value_f32_impl(const char* name_literal, float value) {
  tracy::Profiler::PlotData(name_literal, value);
}

void iree_tracing_plot_value_f64_impl(const char* name_literal, double value) {
  tracy::Profiler::PlotData(name_literal, value);
}

void iree_tracing_mutex_announce(const iree_tracing_location_t* src_loc,
                                 uint32_t* out_lock_id) {
  uint32_t lock_id =
      tracy::GetLockCounter().fetch_add(1, std::memory_order_relaxed);
  assert(lock_id != std::numeric_limits<uint32_t>::max());
  *out_lock_id = lock_id;

  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockAnnounce);
  tracy::MemWrite(&item->lockAnnounce.id, lock_id);
  tracy::MemWrite(&item->lockAnnounce.time, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->lockAnnounce.lckloc,
                  reinterpret_cast<uint64_t>(src_loc));
  tracy::MemWrite(&item->lockAnnounce.type, tracy::LockType::Lockable);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_terminate(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockTerminate);
  tracy::MemWrite(&item->lockTerminate.id, lock_id);
  tracy::MemWrite(&item->lockTerminate.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_before_lock(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockWait);
  tracy::MemWrite(&item->lockWait.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->lockWait.id, lock_id);
  tracy::MemWrite(&item->lockWait.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_after_lock(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockObtain);
  tracy::MemWrite(&item->lockObtain.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->lockObtain.id, lock_id);
  tracy::MemWrite(&item->lockObtain.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_mutex_after_try_lock(uint32_t lock_id, bool was_acquired) {
  if (was_acquired) {
    iree_tracing_mutex_after_lock(lock_id);
  }
}

void iree_tracing_mutex_after_unlock(uint32_t lock_id) {
  auto item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::LockRelease);
  tracy::MemWrite(&item->lockReleaseShared.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->lockRelease.id, lock_id);
  tracy::MemWrite(&item->lockRelease.time, tracy::Profiler::GetTime());
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_register_custom_file_contents(
    tracy_file_mapping const* file_mapping) {
  tracy::Profiler::SourceCallbackRegister(
      [](void* data, const char* filename, size_t& size) -> char* {
        tracy_file_mapping const* file_mapping =
            (tracy_file_mapping const*)data;
        fprintf(stdout, "Tracy source callback queried file: '%s'\n", filename);

        tracy_file_contents* found_file_contents =
            (tracy_file_contents*)bsearch(filename, file_mapping->file_contents,
                                          file_mapping->file_mapping_length,
                                          sizeof(*file_mapping->file_contents),
                                          tracy_file_contents_search_cmp);

        if (!found_file_contents) {
          return nullptr;
        }

        fprintf(stdout, "...Found match: %s\n", found_file_contents->file_name);
        auto buf = (char*)tracy::tracy_malloc_fast(
            found_file_contents->file_contents_length);
        strcpy(buf, found_file_contents->file_contents);
        size = found_file_contents->file_contents_length;
        return buf;
      },
      /*data=*/(void*)file_mapping);
}

int tracy_file_contents_sort_cmp(const void* a, const void* b) {
  tracy_file_contents* file_contents_a = (tracy_file_contents*)a;
  tracy_file_contents* file_contents_b = (tracy_file_contents*)b;
  return strcmp(file_contents_a->file_name, file_contents_b->file_name);
}

int tracy_file_contents_search_cmp(const void* key, const void* val) {
  const char* filename_key = (const char*)key;
  tracy_file_contents* file_contents = (tracy_file_contents*)val;
  return strcmp(filename_key, file_contents->file_name);
}

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

int64_t iree_tracing_time(void) { return tracy::Profiler::GetTime(); }

int64_t iree_tracing_frequency(void) { return tracy::GetFrequencyQpc(); }

uint8_t iree_tracing_gpu_context_allocate(iree_tracing_gpu_context_type_t type,
                                          const char* name, size_t name_length,
                                          bool is_calibrated,
                                          uint64_t cpu_timestamp,
                                          uint64_t gpu_timestamp,
                                          float timestamp_period) {
  // Allocate the process-unique GPU context ID. There's a max of 255 available;
  // if we are recreating devices a lot we may exceed that. Don't do that, or
  // wrap around and get weird (but probably still usable) numbers.
  uint8_t context_id =
      tracy::GetGpuCtxCounter().fetch_add(1, std::memory_order_relaxed);
  if (context_id >= 255) {
    context_id %= 255;
  }

  uint8_t context_flags = 0;
  if (is_calibrated) {
    // Tell tracy we'll be passing calibrated timestamps and not to mess with
    // the times. We'll periodically send GpuCalibration events in case the
    // times drift.
    context_flags |= tracy::GpuContextCalibration;
  }
  {
    auto* item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuNewContext);
    tracy::MemWrite(&item->gpuNewContext.cpuTime, cpu_timestamp);
    tracy::MemWrite(&item->gpuNewContext.gpuTime, gpu_timestamp);
    memset(&item->gpuNewContext.thread, 0, sizeof(item->gpuNewContext.thread));
    tracy::MemWrite(&item->gpuNewContext.period, timestamp_period);
    tracy::MemWrite(&item->gpuNewContext.context, context_id);
    tracy::MemWrite(&item->gpuNewContext.flags, context_flags);
    tracy::MemWrite(&item->gpuNewContext.type, (tracy::GpuContextType)type);
    tracy::Profiler::QueueSerialFinish();
  }

  // Send the name of the context along.
  // NOTE: Tracy will unconditionally free the name so we must clone it here.
  // Since internally Tracy will use its own rpmalloc implementation we must
  // make sure we allocate from the same source.
  char* cloned_name = (char*)tracy::tracy_malloc(name_length);
  memcpy(cloned_name, name, name_length);
  {
    auto* item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuContextName);
    tracy::MemWrite(&item->gpuContextNameFat.context, context_id);
    tracy::MemWrite(&item->gpuContextNameFat.ptr, (uint64_t)cloned_name);
    tracy::MemWrite(&item->gpuContextNameFat.size, name_length);
    tracy::Profiler::QueueSerialFinish();
  }

  return context_id;
}

void iree_tracing_gpu_context_calibrate(uint8_t context_id, int64_t cpu_delta,
                                        int64_t cpu_timestamp,
                                        int64_t gpu_timestamp) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuCalibration);
  tracy::MemWrite(&item->gpuCalibration.gpuTime, gpu_timestamp);
  tracy::MemWrite(&item->gpuCalibration.cpuTime, cpu_timestamp);
  tracy::MemWrite(&item->gpuCalibration.cpuDelta, cpu_delta);
  tracy::MemWrite(&item->gpuCalibration.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_begin(uint8_t context_id, uint16_t query_id,
                                 const iree_tracing_location_t* src_loc) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneBegin.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_begin_external(
    uint8_t context_id, uint16_t query_id, const char* file_name,
    size_t file_name_length, uint32_t line, const char* function_name,
    size_t function_name_length, const char* name, size_t name_length) {
  const auto src_loc = tracy::Profiler::AllocSourceLocation(
      line, file_name, file_name_length, function_name, function_name_length,
      name, name_length);
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type,
                  tracy::QueueType::GpuZoneBeginAllocSrcLocSerial);
  tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneBegin.srcloc, (uint64_t)src_loc);
  tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneBegin.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneBegin.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_end(uint8_t context_id, uint16_t query_id) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
  tracy::MemWrite(&item->gpuZoneEnd.cpuTime, tracy::Profiler::GetTime());
  tracy::MemWrite(&item->gpuZoneEnd.thread, tracy::GetThreadHandle());
  tracy::MemWrite(&item->gpuZoneEnd.queryId, query_id);
  tracy::MemWrite(&item->gpuZoneEnd.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

void iree_tracing_gpu_zone_notify(uint8_t context_id, uint16_t query_id,
                                  int64_t gpu_timestamp) {
  auto* item = tracy::Profiler::QueueSerial();
  tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
  tracy::MemWrite(&item->gpuTime.gpuTime, gpu_timestamp);
  tracy::MemWrite(&item->gpuTime.queryId, query_id);
  tracy::MemWrite(&item->gpuTime.context, context_id);
  tracy::Profiler::QueueSerialFinish();
}

#endif  // IREE_TRACING_FEATURE_INSTRUMENTATION_DEVICE

void* iree_tracing_obscure_ptr(void* ptr) { return ptr; }

#endif  // IREE_TRACING_FEATURES

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
