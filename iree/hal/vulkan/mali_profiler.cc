// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vulkan/mali_profiler.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

#include "third_party/hwcpipe/vendor/arm/mali/mali_profiler.h"

struct iree_hal_vulkan_mali_profiler_context_t {
  hwcpipe::MaliProfiler* mali_profiler;
};

iree_status_t iree_hal_vulkan_mali_profiler_context_allocate(
    iree_allocator_t host_allocator,
    iree_hal_vulkan_mali_profiler_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(out_context);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_context = NULL;

  iree_hal_vulkan_mali_profiler_context_t* context = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*context), (void**)&context);
  if (!iree_status_is_ok(status)) return status;

  hwcpipe::MaliProfiler* profiler = NULL;
  status = iree_allocator_malloc(host_allocator, sizeof(*profiler),
                                 (void**)&profiler);
  if (!iree_status_is_ok(status)) return status;

  // For now just hardcode the list of counters interesting to us.
  hwcpipe::GpuCounterSet enabled_counters{
      hwcpipe::GpuCounter::GpuCycles,
      hwcpipe::GpuCounter::VertexComputeCycles,
      hwcpipe::GpuCounter::VertexComputeJobs,
      hwcpipe::GpuCounter::Instructions,
      hwcpipe::GpuCounter::DivergedInstructions,
      hwcpipe::GpuCounter::ShaderCycles,
      hwcpipe::GpuCounter::ShaderArithmeticCycles,
      hwcpipe::GpuCounter::ShaderLoadStoreCycles,
      hwcpipe::GpuCounter::CacheReadLookups,
      hwcpipe::GpuCounter::CacheWriteLookups,
      hwcpipe::GpuCounter::ExternalMemoryReadAccesses,
      hwcpipe::GpuCounter::ExternalMemoryWriteAccesses,
      hwcpipe::GpuCounter::ExternalMemoryReadStalls,
      hwcpipe::GpuCounter::ExternalMemoryWriteStalls,
      hwcpipe::GpuCounter::ExternalMemoryReadBytes,
      hwcpipe::GpuCounter::ExternalMemoryWriteBytes,
  };

  context->mali_profiler =
      new (profiler) hwcpipe::MaliProfiler(enabled_counters);

  *out_context = context;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_vulkan_mali_profiler_context_free(
    iree_allocator_t host_allocator,
    iree_hal_vulkan_mali_profiler_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  context->mali_profiler->~MaliProfiler();

  iree_allocator_free(host_allocator, context->mali_profiler);
  iree_allocator_free(host_allocator, context);
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_mali_profiler_start(
    iree_hal_vulkan_mali_profiler_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  context->mali_profiler->run();
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_mali_profiler_stop(
    iree_hal_vulkan_mali_profiler_context_t* context) {
  IREE_TRACE_ZONE_BEGIN(z0);
  context->mali_profiler->stop();
  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_mali_profiler_configure_counter(
    iree_hal_vulkan_mali_profiler_context_t* context,
    void (*counter_type_consumer)(const char* counter_name,
                                  uint8_t counter_type)) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (const auto& counter : context->mali_profiler->enabled_counters()) {
    const hwcpipe::GpuCounterInfo& info = hwcpipe::gpu_counter_info.at(counter);
    const std::string& name = info.desc;
    const std::string& unit = info.unit;

    if (unit == "B") {  // Byte
      counter_type_consumer(name.c_str(), IREE_TRACING_PLOT_TYPE_MEMORY);
    } else {
      counter_type_consumer(name.c_str(), IREE_TRACING_PLOT_TYPE_NUMBER);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_vulkan_mali_profiler_sample(
    iree_hal_vulkan_mali_profiler_context_t* context,
    void (*counter_sample_consumer)(const char* counter_name,
                                    int64_t sample_value)) {
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_vulkan_mali_profiler_sample#query");
  const hwcpipe::GpuMeasurements& samples = context->mali_profiler->sample();
  IREE_TRACE_ZONE_END(z0);

  IREE_TRACE_ZONE_BEGIN_NAMED(z1,
                              "iree_hal_vulkan_mali_profiler_sample#upload");
  for (const auto& counter_sample : samples) {
    hwcpipe::GpuCounter counter = counter_sample.first;
    const hwcpipe::Value& sample = counter_sample.second;

    const hwcpipe::GpuCounterInfo& info = hwcpipe::gpu_counter_info.at(counter);
    const std::string& name = info.desc;

    counter_sample_consumer(name.c_str(), sample.get<int64_t>());
  }

  IREE_TRACE_ZONE_END(z1);
}

#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
