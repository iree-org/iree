// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/profile/explain.h"

#include <stdlib.h>
#include <string.h>

#include "iree/tooling/profile/dispatch.h"
#include "iree/tooling/profile/memory.h"
#include "iree/tooling/profile/reader.h"
#include "iree/tooling/profile/summary.h"

#define IREE_PROFILE_EXPLAIN_TOP_EXPORT_COUNT 10

typedef struct iree_profile_explain_export_rank_t {
  // Session-local physical device ordinal for this export aggregate.
  uint32_t physical_device_ordinal;
  // Producer-local executable identifier for this export aggregate.
  uint64_t executable_id;
  // Export ordinal for this export aggregate.
  uint32_t export_ordinal;
  // Total dispatch count for this export aggregate.
  uint64_t dispatch_count;
  // Valid dispatch count for this export aggregate.
  uint64_t valid_count;
  // Invalid dispatch count for this export aggregate.
  uint64_t invalid_count;
  // Maximum valid dispatch duration in raw device ticks.
  uint64_t maximum_ticks;
  // Total valid dispatch duration in raw device ticks.
  double total_ticks;
  // Average valid dispatch duration in nanoseconds when clock fit is available.
  double average_ns;
  // Maximum valid dispatch duration in nanoseconds when clock fit is available.
  double maximum_ns;
  // Total valid dispatch duration in nanoseconds when clock fit is available.
  double total_ns;
  // True when nanosecond values were computed from a device clock fit.
  bool has_clock_fit;
} iree_profile_explain_export_rank_t;

typedef struct iree_profile_explain_interval_t {
  // Inclusive dispatch start tick for this interval.
  uint64_t start_tick;
  // Exclusive dispatch end tick for this interval.
  uint64_t end_tick;
} iree_profile_explain_interval_t;

static double iree_profile_explain_export_rank_score(
    const iree_profile_explain_export_rank_t* rank) {
  return rank->has_clock_fit ? rank->total_ns : rank->total_ticks;
}

static int iree_profile_explain_compare_export_rank(const void* lhs,
                                                    const void* rhs) {
  const iree_profile_explain_export_rank_t* a =
      (const iree_profile_explain_export_rank_t*)lhs;
  const iree_profile_explain_export_rank_t* b =
      (const iree_profile_explain_export_rank_t*)rhs;
  const double a_score = iree_profile_explain_export_rank_score(a);
  const double b_score = iree_profile_explain_export_rank_score(b);
  if (a_score < b_score) return 1;
  if (a_score > b_score) return -1;
  return 0;
}

static int iree_profile_explain_compare_top_event(const void* lhs,
                                                  const void* rhs) {
  const iree_profile_dispatch_top_event_t* a =
      (const iree_profile_dispatch_top_event_t*)lhs;
  const iree_profile_dispatch_top_event_t* b =
      (const iree_profile_dispatch_top_event_t*)rhs;
  if (a->duration_ticks < b->duration_ticks) return 1;
  if (a->duration_ticks > b->duration_ticks) return -1;
  return 0;
}

static int iree_profile_explain_compare_interval(const void* lhs,
                                                 const void* rhs) {
  const iree_profile_explain_interval_t* a =
      (const iree_profile_explain_interval_t*)lhs;
  const iree_profile_explain_interval_t* b =
      (const iree_profile_explain_interval_t*)rhs;
  if (a->start_tick < b->start_tick) return -1;
  if (a->start_tick > b->start_tick) return 1;
  if (a->end_tick < b->end_tick) return -1;
  if (a->end_tick > b->end_tick) return 1;
  return 0;
}

static bool iree_profile_explain_try_fit_driver_host_cpu_clock(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal,
    iree_profile_model_clock_fit_t* out_clock_fit) {
  const iree_profile_model_device_t* device =
      iree_profile_model_find_device(&context->model, physical_device_ordinal);
  return iree_profile_model_device_try_fit_clock_exact(
      device, IREE_PROFILE_MODEL_CLOCK_TIME_DOMAIN_HOST_CPU_TIMESTAMP_NS,
      out_clock_fit);
}

static iree_status_t iree_profile_explain_collect_export_ranks(
    const iree_profile_dispatch_context_t* context,
    iree_allocator_t host_allocator,
    iree_profile_explain_export_rank_t** out_ranks,
    iree_host_size_t* out_rank_count) {
  *out_ranks = NULL;
  *out_rank_count = 0;
  if (context->aggregate_count == 0) return iree_ok_status();

  iree_profile_explain_export_rank_t* ranks = NULL;
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      host_allocator, context->aggregate_count, sizeof(ranks[0]),
      (void**)&ranks);

  iree_host_size_t rank_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
      const iree_profile_dispatch_aggregate_t* aggregate =
          &context->aggregates[i];
      if (aggregate->valid_count == 0) continue;

      const iree_profile_model_device_t* device =
          iree_profile_model_find_device(&context->model,
                                         aggregate->physical_device_ordinal);
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_model_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      (void)tick_frequency_hz;

      iree_profile_explain_export_rank_t* rank = &ranks[rank_count++];
      memset(rank, 0, sizeof(*rank));
      rank->physical_device_ordinal = aggregate->physical_device_ordinal;
      rank->executable_id = aggregate->executable_id;
      rank->export_ordinal = aggregate->export_ordinal;
      rank->dispatch_count = aggregate->dispatch_count;
      rank->valid_count = aggregate->valid_count;
      rank->invalid_count = aggregate->invalid_count;
      rank->maximum_ticks = aggregate->maximum_ticks;
      rank->total_ticks = aggregate->total_ticks;
      rank->has_clock_fit = has_clock_fit;
      if (has_clock_fit) {
        rank->average_ns = aggregate->mean_ticks * ns_per_tick;
        rank->maximum_ns = (double)aggregate->maximum_ticks * ns_per_tick;
        rank->total_ns = aggregate->total_ticks * ns_per_tick;
      }
    }
  }

  if (iree_status_is_ok(status)) {
    qsort(ranks, rank_count, sizeof(ranks[0]),
          iree_profile_explain_compare_export_rank);
    *out_ranks = ranks;
    *out_rank_count = rank_count;
  } else {
    iree_allocator_free(host_allocator, ranks);
  }
  return status;
}

static void iree_profile_explain_accumulate_queue_operation_totals(
    const iree_profile_dispatch_context_t* context, uint64_t event_counts[],
    uint64_t payload_bytes[], uint64_t strategy_counts[]) {
  for (iree_host_size_t i = 0; i < context->model.queue_event_count; ++i) {
    const iree_hal_profile_queue_event_t* event =
        &context->model.queue_events[i].record;
    if (event->type <= IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL) {
      ++event_counts[event->type];
      payload_bytes[event->type] += event->payload_length;
    }
    if (event->dependency_strategy <=
        IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER) {
      ++strategy_counts[event->dependency_strategy];
    }
  }
}

static double iree_profile_explain_visible_span_ticks(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal) {
  uint64_t earliest_start_tick = UINT64_MAX;
  uint64_t latest_end_tick = 0;
  for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
    const iree_profile_dispatch_queue_aggregate_t* aggregate =
        &context->queue_aggregates[i];
    if (aggregate->physical_device_ordinal != physical_device_ordinal ||
        aggregate->valid_count == 0) {
      continue;
    }
    earliest_start_tick =
        iree_min(earliest_start_tick, aggregate->earliest_start_tick);
    latest_end_tick = iree_max(latest_end_tick, aggregate->latest_end_tick);
  }
  return iree_profile_model_span_ticks(earliest_start_tick, latest_end_tick);
}

static double iree_profile_explain_total_dispatch_ticks_for_device(
    const iree_profile_dispatch_context_t* context,
    uint32_t physical_device_ordinal) {
  double total_ticks = 0.0;
  for (iree_host_size_t i = 0; i < context->aggregate_count; ++i) {
    const iree_profile_dispatch_aggregate_t* aggregate =
        &context->aggregates[i];
    if (aggregate->physical_device_ordinal == physical_device_ordinal) {
      total_ticks += aggregate->total_ticks;
    }
  }
  return total_ticks;
}

static iree_status_t iree_profile_explain_summarize_queue(
    const iree_profile_dispatch_context_t* context,
    const iree_hal_profile_queue_record_t* queue,
    iree_allocator_t host_allocator, uint64_t* out_submission_count,
    uint64_t* out_valid_submission_count,
    uint64_t* out_invalid_submission_count, double* out_busy_ticks,
    double* out_total_dispatch_ticks, uint64_t* out_gap_count,
    double* out_total_gap_ticks, double* out_max_gap_ticks) {
  *out_submission_count = 0;
  *out_valid_submission_count = 0;
  *out_invalid_submission_count = 0;
  *out_busy_ticks = 0.0;
  *out_total_dispatch_ticks = 0.0;
  *out_gap_count = 0;
  *out_total_gap_ticks = 0.0;
  *out_max_gap_ticks = 0.0;
  if (context->queue_aggregate_count == 0) return iree_ok_status();

  iree_profile_explain_interval_t* intervals = NULL;
  iree_status_t status = iree_allocator_malloc_array_uninitialized(
      host_allocator, context->queue_aggregate_count, sizeof(intervals[0]),
      (void**)&intervals);

  iree_host_size_t interval_count = 0;
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < context->queue_aggregate_count; ++i) {
      const iree_profile_dispatch_queue_aggregate_t* aggregate =
          &context->queue_aggregates[i];
      if (aggregate->physical_device_ordinal !=
              queue->physical_device_ordinal ||
          aggregate->queue_ordinal != queue->queue_ordinal ||
          aggregate->stream_id != queue->stream_id) {
        continue;
      }
      *out_submission_count += 1;
      *out_total_dispatch_ticks += aggregate->total_ticks;
      if (aggregate->valid_count != 0) {
        *out_valid_submission_count += 1;
        intervals[interval_count++] = (iree_profile_explain_interval_t){
            aggregate->earliest_start_tick, aggregate->latest_end_tick};
      }
      if (aggregate->invalid_count != 0) {
        *out_invalid_submission_count += 1;
      }
    }
  }

  if (iree_status_is_ok(status) && interval_count != 0) {
    qsort(intervals, interval_count, sizeof(intervals[0]),
          iree_profile_explain_compare_interval);

    uint64_t merged_start_tick = intervals[0].start_tick;
    uint64_t merged_end_tick = intervals[0].end_tick;
    for (iree_host_size_t i = 1; i < interval_count; ++i) {
      if (intervals[i].start_tick <= merged_end_tick) {
        merged_end_tick = iree_max(merged_end_tick, intervals[i].end_tick);
        continue;
      }
      *out_busy_ticks += (double)(merged_end_tick - merged_start_tick);
      const uint64_t gap_ticks = intervals[i].start_tick - merged_end_tick;
      ++*out_gap_count;
      *out_total_gap_ticks += (double)gap_ticks;
      *out_max_gap_ticks = iree_max(*out_max_gap_ticks, (double)gap_ticks);
      merged_start_tick = intervals[i].start_tick;
      merged_end_tick = intervals[i].end_tick;
    }
    *out_busy_ticks += (double)(merged_end_tick - merged_start_tick);
  }

  iree_allocator_free(host_allocator, intervals);
  return status;
}

static void iree_profile_explain_print_hint_text(const char* severity,
                                                 const char* category,
                                                 const char* message,
                                                 FILE* file) {
  fprintf(file, "  [%s] %s: %s\n", severity, category, message);
}

static void iree_profile_explain_print_hint_jsonl(const char* severity,
                                                  const char* category,
                                                  const char* message,
                                                  FILE* file) {
  fprintf(file, "{\"type\":\"explain_hint\",\"severity\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(severity));
  fprintf(file, ",\"category\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(category));
  fprintf(file, ",\"message\":");
  iree_profile_fprint_json_string(file, iree_make_cstring_view(message));
  fputs("}\n", file);
}

static iree_status_t iree_profile_explain_print_text(
    const iree_profile_summary_t* summary,
    const iree_profile_dispatch_context_t* dispatch_context,
    const iree_profile_memory_context_t* memory_context,
    iree_allocator_t host_allocator, FILE* file) {
  iree_profile_explain_export_rank_t* export_ranks = NULL;
  iree_host_size_t export_rank_count = 0;
  iree_status_t status = iree_profile_explain_collect_export_ranks(
      dispatch_context, host_allocator, &export_ranks, &export_rank_count);

  if (iree_status_is_ok(status)) {
    fprintf(file, "IREE HAL profile explain\n");
    fprintf(file,
            "health: records=%" PRIu64 " chunks=%" PRIu64
            " unknown_records=%" PRIu64 " unknown_chunks=%" PRIu64
            " truncated_chunks=%" PRIu64 "\n",
            summary->file_record_count, summary->chunk_count,
            summary->unknown_record_count, summary->unknown_chunk_count,
            summary->truncated_chunk_count);
    fprintf(file,
            "coverage: dispatches=%" PRIu64 " valid=%" PRIu64
            " invalid=%" PRIu64 " queues=%" PRIhsz " queue_events=%" PRIhsz
            " command_buffers=%" PRIhsz " command_operations=%" PRIhsz
            " memory_events=%" PRIu64 "\n",
            dispatch_context->matched_dispatch_count,
            dispatch_context->valid_dispatch_count,
            dispatch_context->invalid_dispatch_count,
            dispatch_context->model.queue_count,
            dispatch_context->model.queue_event_count,
            dispatch_context->model.command_buffer_count,
            dispatch_context->model.command_operation_count,
            memory_context->matched_event_count);

    fprintf(file, "devices:\n");
    for (iree_host_size_t i = 0; i < dispatch_context->model.device_count;
         ++i) {
      const iree_profile_model_device_t* device =
          &dispatch_context->model.devices[i];
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_model_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double visible_span_ticks = iree_profile_explain_visible_span_ticks(
          dispatch_context, device->physical_device_ordinal);
      const double total_dispatch_ticks =
          iree_profile_explain_total_dispatch_ticks_for_device(
              dispatch_context, device->physical_device_ordinal);
      const double active_ratio =
          visible_span_ticks > 0.0 ? total_dispatch_ticks / visible_span_ticks
                                   : 0.0;
      fprintf(file,
              "  device[%u]: clock_fit=%s clock_samples=%" PRIu64
              " visible_span_ticks=%.3f active_dispatch_ticks=%.3f"
              " active_over_visible=%.3f\n",
              device->physical_device_ordinal, has_clock_fit ? "true" : "false",
              device->clock_sample_count, visible_span_ticks,
              total_dispatch_ticks, active_ratio);
      if (has_clock_fit) {
        fprintf(file,
                "    visible_span_ns=%.3f active_dispatch_ns=%.3f"
                " tick_frequency_hz=%.3f\n",
                visible_span_ticks * ns_per_tick,
                total_dispatch_ticks * ns_per_tick, tick_frequency_hz);
      }
    }

    fprintf(file, "queues:\n");
    for (iree_host_size_t i = 0;
         i < dispatch_context->model.queue_count && iree_status_is_ok(status);
         ++i) {
      const iree_hal_profile_queue_record_t* queue =
          &dispatch_context->model.queues[i].record;
      uint64_t submission_count = 0;
      uint64_t valid_submission_count = 0;
      uint64_t invalid_submission_count = 0;
      double busy_ticks = 0.0;
      double total_dispatch_ticks = 0.0;
      uint64_t gap_count = 0;
      double total_gap_ticks = 0.0;
      double max_gap_ticks = 0.0;
      status = iree_profile_explain_summarize_queue(
          dispatch_context, queue, host_allocator, &submission_count,
          &valid_submission_count, &invalid_submission_count, &busy_ticks,
          &total_dispatch_ticks, &gap_count, &total_gap_ticks, &max_gap_ticks);
      if (iree_status_is_ok(status)) {
        const iree_profile_model_device_t* device =
            iree_profile_model_find_device(&dispatch_context->model,
                                           queue->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_model_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        (void)tick_frequency_hz;
        fprintf(file,
                "  queue device=%u ordinal=%u stream=%" PRIu64
                ": submissions=%" PRIu64 " valid=%" PRIu64 " invalid=%" PRIu64
                " busy_ticks=%.3f"
                " total_dispatch_ticks=%.3f",
                queue->physical_device_ordinal, queue->queue_ordinal,
                queue->stream_id, submission_count, valid_submission_count,
                invalid_submission_count, busy_ticks, total_dispatch_ticks);
        if (has_clock_fit) {
          fprintf(file, " busy_ns=%.3f total_dispatch_ns=%.3f",
                  busy_ticks * ns_per_tick, total_dispatch_ticks * ns_per_tick);
        }
        if (dispatch_context->model.queue_event_count != 0) {
          fprintf(file,
                  " gaps=%" PRIu64 " total_gap_ticks=%.3f max_gap_ticks=%.3f",
                  gap_count, total_gap_ticks, max_gap_ticks);
          if (has_clock_fit) {
            fprintf(file, " total_gap_ns=%.3f max_gap_ns=%.3f",
                    total_gap_ticks * ns_per_tick, max_gap_ticks * ns_per_tick);
          }
        } else {
          fprintf(file, " gaps=unavailable_without_queue_events");
        }
        fputc('\n', file);
      }
    }

    fprintf(file, "top exports by total dispatch time:\n");
    const iree_host_size_t top_export_count =
        iree_min(export_rank_count,
                 (iree_host_size_t)IREE_PROFILE_EXPLAIN_TOP_EXPORT_COUNT);
    for (iree_host_size_t i = 0;
         i < top_export_count && iree_status_is_ok(status); ++i) {
      const iree_profile_explain_export_rank_t* rank = &export_ranks[i];
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_model_resolve_dispatch_key(
          &dispatch_context->model, rank->physical_device_ordinal,
          rank->executable_id, rank->export_ordinal, numeric_buffer,
          sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "  #%" PRIhsz " %.*s device=%u executable=%" PRIu64
                " export=%u count=%" PRIu64 " valid=%" PRIu64
                " invalid=%" PRIu64 " total_ticks=%.3f max_ticks=%" PRIu64,
                i + 1, (int)key.size, key.data, rank->physical_device_ordinal,
                rank->executable_id, rank->export_ordinal, rank->dispatch_count,
                rank->valid_count, rank->invalid_count, rank->total_ticks,
                rank->maximum_ticks);
        if (rank->has_clock_fit) {
          fprintf(file, " total_ns=%.3f avg_ns=%.3f max_ns=%.3f",
                  rank->total_ns, rank->average_ns, rank->maximum_ns);
        }
        fputc('\n', file);
      }
    }

    iree_profile_dispatch_top_event_t
        top_dispatches[IREE_PROFILE_DISPATCH_TOP_EVENT_COUNT];
    memcpy(top_dispatches, dispatch_context->top_dispatches,
           dispatch_context->top_dispatch_count * sizeof(top_dispatches[0]));
    qsort(top_dispatches, dispatch_context->top_dispatch_count,
          sizeof(top_dispatches[0]), iree_profile_explain_compare_top_event);
    fprintf(file, "top individual dispatches:\n");
    for (iree_host_size_t i = 0;
         i < dispatch_context->top_dispatch_count && iree_status_is_ok(status);
         ++i) {
      const iree_profile_dispatch_top_event_t* top_event = &top_dispatches[i];
      iree_profile_model_clock_fit_t clock_fit;
      const bool has_clock_fit =
          iree_profile_explain_try_fit_driver_host_cpu_clock(
              dispatch_context, top_event->physical_device_ordinal, &clock_fit);
      int64_t duration_ns = 0;
      const bool has_duration_ns =
          has_clock_fit &&
          iree_profile_model_clock_fit_scale_ticks_to_ns(
              &clock_fit, top_event->duration_ticks, &duration_ns);
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_model_resolve_dispatch_key(
          &dispatch_context->model, top_event->physical_device_ordinal,
          top_event->event.executable_id, top_event->event.export_ordinal,
          numeric_buffer, sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "  #%" PRIhsz " event=%" PRIu64 " submission=%" PRIu64
                " %.*s device=%u queue=%u stream=%" PRIu64
                " duration_ticks=%" PRIu64,
                i + 1, top_event->event.event_id,
                top_event->event.submission_id, (int)key.size, key.data,
                top_event->physical_device_ordinal, top_event->queue_ordinal,
                top_event->stream_id, top_event->duration_ticks);
        if (has_duration_ns) {
          fprintf(file, " duration_ns=%" PRId64, duration_ns);
        }
        fputc('\n', file);
      }
    }

    uint64_t event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t strategy_counts
        [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER + 1] = {0};
    iree_profile_explain_accumulate_queue_operation_totals(
        dispatch_context, event_counts, payload_bytes, strategy_counts);
    fprintf(file,
            "queue operations: events=%" PRIhsz
            " strategies none/inline/device_barrier/software_defer=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            dispatch_context->model.queue_event_count,
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE],
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER]);
    fprintf(file,
            "  transfer_payload_bytes copy/fill/update/read/write=%" PRIu64
            "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE]);
    fprintf(
        file,
        "  operation_counts dispatch/execute/alloca/dealloca/host_call=%" PRIu64
        "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA],
        event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL]);

    fprintf(file, "memory pressure:\n");
    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      const iree_profile_memory_device_t* device = &memory_context->devices[i];
      fprintf(file,
              "  device[%u]: slab_high_water=%" PRIu64
              " pool_high_water=%" PRIu64 " queue_high_water=%" PRIu64
              " buffer_high_water=%" PRIu64 " pool_waits=%" PRIu64 "\n",
              device->physical_device_ordinal, device->high_water_slab_bytes,
              device->high_water_pool_reserved_bytes,
              device->high_water_queue_bytes, device->high_water_buffer_bytes,
              device->pool_wait_count);
    }

    fprintf(file, "hints:\n");
    if (summary->unknown_record_count != 0 ||
        summary->unknown_chunk_count != 0) {
      iree_profile_explain_print_hint_text(
          "warning", "format",
          "unknown records or chunks were present; newer producer data may be "
          "missing from this analysis",
          file);
    }
    if (summary->truncated_chunk_count != 0) {
      iree_profile_explain_print_hint_text(
          "error", "format",
          "truncated chunks were present; timing and lifecycle totals may be "
          "incomplete",
          file);
    }
    if (dispatch_context->invalid_dispatch_count != 0) {
      iree_profile_explain_print_hint_text(
          "warning", "dispatch",
          "some dispatch events had missing or reversed timestamps and were "
          "excluded from timing totals",
          file);
    }
    if (dispatch_context->model.queue_event_count == 0) {
      iree_profile_explain_print_hint_text(
          "info", "queue",
          "queue event records are absent; queue dependency and gap hints are "
          "limited to dispatch timestamp intervals",
          file);
    }
    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      if (memory_context->devices[i].pool_wait_count != 0) {
        iree_profile_explain_print_hint_text(
            "warning", "memory",
            "pool wait events were observed; allocation readiness affected the "
            "captured queue schedule",
            file);
        break;
      }
    }
    if (dispatch_context->valid_dispatch_count == 0) {
      iree_profile_explain_print_hint_text(
          "info", "dispatch",
          "no valid dispatch events were captured; enable queue profiling on a "
          "producer that emits device timestamps",
          file);
    }
  }

  iree_allocator_free(host_allocator, export_ranks);
  return status;
}

static iree_status_t iree_profile_explain_print_jsonl(
    const iree_profile_summary_t* summary,
    const iree_profile_dispatch_context_t* dispatch_context,
    const iree_profile_memory_context_t* memory_context,
    iree_allocator_t host_allocator, FILE* file) {
  iree_profile_explain_export_rank_t* export_ranks = NULL;
  iree_host_size_t export_rank_count = 0;
  iree_status_t status = iree_profile_explain_collect_export_ranks(
      dispatch_context, host_allocator, &export_ranks, &export_rank_count);

  if (iree_status_is_ok(status)) {
    fprintf(file,
            "{\"type\":\"explain_summary\",\"file_records\":%" PRIu64
            ",\"chunk_records\":%" PRIu64 ",\"unknown_records\":%" PRIu64
            ",\"unknown_chunks\":%" PRIu64 ",\"truncated_chunks\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"valid_dispatches\":%" PRIu64
            ",\"invalid_dispatches\":%" PRIu64 ",\"queues\":%" PRIhsz
            ",\"queue_events\":%" PRIhsz ",\"command_buffers\":%" PRIhsz
            ",\"command_operations\":%" PRIhsz ",\"memory_events\":%" PRIu64
            "}\n",
            summary->file_record_count, summary->chunk_count,
            summary->unknown_record_count, summary->unknown_chunk_count,
            summary->truncated_chunk_count,
            dispatch_context->matched_dispatch_count,
            dispatch_context->valid_dispatch_count,
            dispatch_context->invalid_dispatch_count,
            dispatch_context->model.queue_count,
            dispatch_context->model.queue_event_count,
            dispatch_context->model.command_buffer_count,
            dispatch_context->model.command_operation_count,
            memory_context->matched_event_count);

    for (iree_host_size_t i = 0; i < dispatch_context->model.device_count;
         ++i) {
      const iree_profile_model_device_t* device =
          &dispatch_context->model.devices[i];
      double ns_per_tick = 0.0;
      double tick_frequency_hz = 0.0;
      const bool has_clock_fit = iree_profile_model_device_try_fit_clock(
          device, &ns_per_tick, &tick_frequency_hz);
      const double visible_span_ticks = iree_profile_explain_visible_span_ticks(
          dispatch_context, device->physical_device_ordinal);
      const double total_dispatch_ticks =
          iree_profile_explain_total_dispatch_ticks_for_device(
              dispatch_context, device->physical_device_ordinal);
      fprintf(
          file,
          "{\"type\":\"explain_device\",\"physical_device_ordinal\":%u"
          ",\"clock_fit_available\":%s,\"clock_samples\":%" PRIu64
          ",\"visible_span_ticks\":%.3f"
          ",\"active_dispatch_ticks\":%.3f"
          ",\"active_over_visible\":%.6f"
          ",\"tick_frequency_hz\":%.3f,\"visible_span_ns\":%.3f"
          ",\"active_dispatch_ns\":%.3f}\n",
          device->physical_device_ordinal, has_clock_fit ? "true" : "false",
          device->clock_sample_count, visible_span_ticks, total_dispatch_ticks,
          visible_span_ticks > 0.0 ? total_dispatch_ticks / visible_span_ticks
                                   : 0.0,
          has_clock_fit ? tick_frequency_hz : 0.0,
          has_clock_fit ? visible_span_ticks * ns_per_tick : 0.0,
          has_clock_fit ? total_dispatch_ticks * ns_per_tick : 0.0);
    }

    for (iree_host_size_t i = 0;
         i < dispatch_context->model.queue_count && iree_status_is_ok(status);
         ++i) {
      const iree_hal_profile_queue_record_t* queue =
          &dispatch_context->model.queues[i].record;
      uint64_t submission_count = 0;
      uint64_t valid_submission_count = 0;
      uint64_t invalid_submission_count = 0;
      double busy_ticks = 0.0;
      double total_dispatch_ticks = 0.0;
      uint64_t gap_count = 0;
      double total_gap_ticks = 0.0;
      double max_gap_ticks = 0.0;
      status = iree_profile_explain_summarize_queue(
          dispatch_context, queue, host_allocator, &submission_count,
          &valid_submission_count, &invalid_submission_count, &busy_ticks,
          &total_dispatch_ticks, &gap_count, &total_gap_ticks, &max_gap_ticks);
      if (iree_status_is_ok(status)) {
        const iree_profile_model_device_t* device =
            iree_profile_model_find_device(&dispatch_context->model,
                                           queue->physical_device_ordinal);
        double ns_per_tick = 0.0;
        double tick_frequency_hz = 0.0;
        const bool has_clock_fit = iree_profile_model_device_try_fit_clock(
            device, &ns_per_tick, &tick_frequency_hz);
        (void)tick_frequency_hz;
        const bool gap_analysis_available =
            dispatch_context->model.queue_event_count != 0;
        fprintf(file,
                "{\"type\":\"explain_queue\",\"physical_device_ordinal\":%u"
                ",\"queue_ordinal\":%u,\"stream_id\":%" PRIu64
                ",\"submissions\":%" PRIu64 ",\"valid_submissions\":%" PRIu64
                ",\"invalid_submissions\":%" PRIu64
                ",\"busy_ticks\":%.3f,\"total_dispatch_ticks\":%.3f"
                ",\"clock_fit_available\":%s,\"busy_ns\":%.3f"
                ",\"total_dispatch_ns\":%.3f"
                ",\"gap_analysis_available\":%s,\"gaps\":%" PRIu64
                ",\"total_gap_ticks\":%.3f,\"max_gap_ticks\":%.3f"
                ",\"total_gap_ns\":%.3f,\"max_gap_ns\":%.3f}\n",
                queue->physical_device_ordinal, queue->queue_ordinal,
                queue->stream_id, submission_count, valid_submission_count,
                invalid_submission_count, busy_ticks, total_dispatch_ticks,
                has_clock_fit ? "true" : "false",
                has_clock_fit ? busy_ticks * ns_per_tick : 0.0,
                has_clock_fit ? total_dispatch_ticks * ns_per_tick : 0.0,
                gap_analysis_available ? "true" : "false",
                gap_analysis_available ? gap_count : 0,
                gap_analysis_available ? total_gap_ticks : 0.0,
                gap_analysis_available ? max_gap_ticks : 0.0,
                gap_analysis_available && has_clock_fit
                    ? total_gap_ticks * ns_per_tick
                    : 0.0,
                gap_analysis_available && has_clock_fit
                    ? max_gap_ticks * ns_per_tick
                    : 0.0);
      }
    }

    const iree_host_size_t top_export_count =
        iree_min(export_rank_count,
                 (iree_host_size_t)IREE_PROFILE_EXPLAIN_TOP_EXPORT_COUNT);
    for (iree_host_size_t i = 0;
         i < top_export_count && iree_status_is_ok(status); ++i) {
      const iree_profile_explain_export_rank_t* rank = &export_ranks[i];
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_model_resolve_dispatch_key(
          &dispatch_context->model, rank->physical_device_ordinal,
          rank->executable_id, rank->export_ordinal, numeric_buffer,
          sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "{\"type\":\"explain_top_export\",\"rank\":%" PRIhsz
                ",\"physical_device_ordinal\":%u"
                ",\"executable_id\":%" PRIu64
                ",\"export_ordinal\":%u"
                ",\"key\":",
                i + 1, rank->physical_device_ordinal, rank->executable_id,
                rank->export_ordinal);
        iree_profile_fprint_json_string(file, key);
        fprintf(file,
                ",\"dispatches\":%" PRIu64 ",\"valid\":%" PRIu64
                ",\"invalid\":%" PRIu64
                ",\"total_ticks\":%.3f"
                ",\"max_ticks\":%" PRIu64
                ",\"clock_fit_available\":%s"
                ",\"total_ns\":%.3f,\"avg_ns\":%.3f,\"max_ns\":%.3f}\n",
                rank->dispatch_count, rank->valid_count, rank->invalid_count,
                rank->total_ticks, rank->maximum_ticks,
                rank->has_clock_fit ? "true" : "false", rank->total_ns,
                rank->average_ns, rank->maximum_ns);
      }
    }

    iree_profile_dispatch_top_event_t
        top_dispatches[IREE_PROFILE_DISPATCH_TOP_EVENT_COUNT];
    memcpy(top_dispatches, dispatch_context->top_dispatches,
           dispatch_context->top_dispatch_count * sizeof(top_dispatches[0]));
    qsort(top_dispatches, dispatch_context->top_dispatch_count,
          sizeof(top_dispatches[0]), iree_profile_explain_compare_top_event);
    for (iree_host_size_t i = 0;
         i < dispatch_context->top_dispatch_count && iree_status_is_ok(status);
         ++i) {
      const iree_profile_dispatch_top_event_t* top_event = &top_dispatches[i];
      iree_profile_model_clock_fit_t clock_fit;
      const bool has_clock_fit =
          iree_profile_explain_try_fit_driver_host_cpu_clock(
              dispatch_context, top_event->physical_device_ordinal, &clock_fit);
      int64_t duration_ns = 0;
      const bool has_duration_ns =
          has_clock_fit &&
          iree_profile_model_clock_fit_scale_ticks_to_ns(
              &clock_fit, top_event->duration_ticks, &duration_ns);
      char numeric_buffer[128];
      iree_string_view_t key = iree_string_view_empty();
      status = iree_profile_model_resolve_dispatch_key(
          &dispatch_context->model, top_event->physical_device_ordinal,
          top_event->event.executable_id, top_event->event.export_ordinal,
          numeric_buffer, sizeof(numeric_buffer), &key);
      if (iree_status_is_ok(status)) {
        fprintf(file,
                "{\"type\":\"explain_top_dispatch\",\"rank\":%" PRIhsz
                ",\"event_id\":%" PRIu64 ",\"submission_id\":%" PRIu64
                ",\"physical_device_ordinal\":%u,\"queue_ordinal\":%u"
                ",\"stream_id\":%" PRIu64 ",\"key\":",
                i + 1, top_event->event.event_id,
                top_event->event.submission_id,
                top_event->physical_device_ordinal, top_event->queue_ordinal,
                top_event->stream_id);
        iree_profile_fprint_json_string(file, key);
        fprintf(file,
                ",\"duration_ticks\":%" PRIu64
                ",\"clock_fit_available\":%s,\"duration_ns\":%" PRId64 "}\n",
                top_event->duration_ticks, has_clock_fit ? "true" : "false",
                has_duration_ns ? duration_ns : 0);
      }
    }

    uint64_t event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL + 1] = {
        0};
    uint64_t strategy_counts
        [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER + 1] = {0};
    iree_profile_explain_accumulate_queue_operation_totals(
        dispatch_context, event_counts, payload_bytes, strategy_counts);
    fprintf(file,
            "{\"type\":\"explain_queue_operations\""
            ",\"events\":%" PRIhsz ",\"strategy_none\":%" PRIu64
            ",\"strategy_inline\":%" PRIu64
            ",\"strategy_device_barrier\":%" PRIu64
            ",\"strategy_software_defer\":%" PRIu64 ",\"copy_bytes\":%" PRIu64
            ",\"fill_bytes\":%" PRIu64 ",\"update_bytes\":%" PRIu64
            ",\"read_bytes\":%" PRIu64 ",\"write_bytes\":%" PRIu64
            ",\"dispatches\":%" PRIu64 ",\"executes\":%" PRIu64
            ",\"allocas\":%" PRIu64 ",\"deallocas\":%" PRIu64
            ",\"host_calls\":%" PRIu64 "}\n",
            dispatch_context->model.queue_event_count,
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_NONE],
            strategy_counts[IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_INLINE],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_DEVICE_BARRIER],
            strategy_counts
                [IREE_HAL_PROFILE_QUEUE_DEPENDENCY_STRATEGY_SOFTWARE_DEFER],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_COPY],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_FILL],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_UPDATE],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ],
            payload_bytes[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_ALLOCA],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DEALLOCA],
            event_counts[IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_HOST_CALL]);

    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      const iree_profile_memory_device_t* device = &memory_context->devices[i];
      fprintf(file,
              "{\"type\":\"explain_memory_pressure\""
              ",\"physical_device_ordinal\":%u"
              ",\"slab_high_water_bytes\":%" PRIu64
              ",\"pool_high_water_bytes\":%" PRIu64
              ",\"queue_high_water_bytes\":%" PRIu64
              ",\"buffer_high_water_bytes\":%" PRIu64 ",\"pool_waits\":%" PRIu64
              "}\n",
              device->physical_device_ordinal, device->high_water_slab_bytes,
              device->high_water_pool_reserved_bytes,
              device->high_water_queue_bytes, device->high_water_buffer_bytes,
              device->pool_wait_count);
    }

    if (summary->unknown_record_count != 0 ||
        summary->unknown_chunk_count != 0) {
      iree_profile_explain_print_hint_jsonl(
          "warning", "format",
          "unknown records or chunks were present; newer producer data may be "
          "missing from this analysis",
          file);
    }
    if (summary->truncated_chunk_count != 0) {
      iree_profile_explain_print_hint_jsonl(
          "error", "format",
          "truncated chunks were present; timing and lifecycle totals may be "
          "incomplete",
          file);
    }
    if (dispatch_context->invalid_dispatch_count != 0) {
      iree_profile_explain_print_hint_jsonl(
          "warning", "dispatch",
          "some dispatch events had missing or reversed timestamps and were "
          "excluded from timing totals",
          file);
    }
    if (dispatch_context->model.queue_event_count == 0) {
      iree_profile_explain_print_hint_jsonl(
          "info", "queue",
          "queue event records are absent; queue dependency and gap hints are "
          "limited to dispatch timestamp intervals",
          file);
    }
    for (iree_host_size_t i = 0; i < memory_context->device_count; ++i) {
      if (memory_context->devices[i].pool_wait_count != 0) {
        iree_profile_explain_print_hint_jsonl(
            "warning", "memory",
            "pool wait events were observed; allocation readiness affected the "
            "captured queue schedule",
            file);
        break;
      }
    }
    if (dispatch_context->valid_dispatch_count == 0) {
      iree_profile_explain_print_hint_jsonl(
          "info", "dispatch",
          "no valid dispatch events were captured; enable queue profiling on a "
          "producer that emits device timestamps",
          file);
    }
  }

  iree_allocator_free(host_allocator, export_ranks);
  return status;
}

typedef struct iree_profile_explain_parse_context_t {
  // File-level summary populated during the metadata pass.
  iree_profile_summary_t* summary;
  // Dispatch model state and event aggregates.
  iree_profile_dispatch_context_t* dispatch_context;
  // Memory event aggregates.
  iree_profile_memory_context_t* memory_context;
  // Optional glob filter applied to dispatch/event keys.
  iree_string_view_t filter;
  // Optional entity identifier filter, or -1 when disabled.
  int64_t id_filter;
  // Output stream passed to shared parsers that support raw event emission.
  FILE* file;
} iree_profile_explain_parse_context_t;

static iree_status_t iree_profile_explain_metadata_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_explain_parse_context_t* context =
      (iree_profile_explain_parse_context_t*)user_data;
  IREE_RETURN_IF_ERROR(
      iree_profile_summary_process_record(context->summary, record));
  return iree_profile_model_process_metadata_record(
      &context->dispatch_context->model, record);
}

static iree_status_t iree_profile_explain_event_record(
    void* user_data, const iree_hal_profile_file_record_t* record,
    iree_host_size_t record_index) {
  (void)record_index;
  iree_profile_explain_parse_context_t* context =
      (iree_profile_explain_parse_context_t*)user_data;
  IREE_RETURN_IF_ERROR(iree_profile_dispatch_process_events_record(
      context->dispatch_context, record, context->filter,
      IREE_PROFILE_PROJECTION_MODE_DISPATCH, context->id_filter,
      /*emit_events=*/false, context->file));
  if (record->header.record_type == IREE_HAL_PROFILE_FILE_RECORD_TYPE_CHUNK &&
      iree_string_view_equal(record->content_type,
                             IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS)) {
    IREE_RETURN_IF_ERROR(iree_profile_model_process_queue_event_records(
        &context->dispatch_context->model, record, IREE_SV("*"),
        /*id_filter=*/-1));
  }
  return iree_profile_memory_context_accumulate_record(
      context->memory_context, record, IREE_SV("*"), /*id_filter=*/-1,
      /*emit_events=*/false, context->file);
}

iree_status_t iree_profile_explain_file(iree_string_view_t path,
                                        iree_string_view_t format,
                                        iree_string_view_t filter,
                                        int64_t id_filter, FILE* file,
                                        iree_allocator_t host_allocator) {
  bool is_text = iree_string_view_equal(format, IREE_SV("text"));
  bool is_jsonl = iree_string_view_equal(format, IREE_SV("jsonl"));
  if (!is_text && !is_jsonl) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported profile output format '%.*s'",
                            (int)format.size, format.data);
  }

  iree_profile_file_t profile_file;
  IREE_RETURN_IF_ERROR(
      iree_profile_file_open(path, host_allocator, &profile_file));

  iree_profile_summary_t summary;
  iree_profile_summary_initialize(host_allocator, &summary);
  iree_profile_dispatch_context_t dispatch_context;
  iree_profile_dispatch_context_initialize(host_allocator, &dispatch_context);
  iree_profile_memory_context_t memory_context;
  iree_profile_memory_context_initialize(host_allocator, &memory_context);
  iree_profile_explain_parse_context_t parse_context = {
      .summary = &summary,
      .dispatch_context = &dispatch_context,
      .memory_context = &memory_context,
      .filter = filter,
      .id_filter = id_filter,
      .file = file,
  };
  iree_profile_file_record_callback_t record_callback = {
      .fn = iree_profile_explain_metadata_record,
      .user_data = &parse_context,
  };
  iree_status_t status =
      iree_profile_file_for_each_record(&profile_file, record_callback);
  if (iree_status_is_ok(status)) {
    record_callback.fn = iree_profile_explain_event_record;
    status = iree_profile_file_for_each_record(&profile_file, record_callback);
  }

  if (iree_status_is_ok(status)) {
    if (is_text) {
      status = iree_profile_explain_print_text(
          &summary, &dispatch_context, &memory_context, host_allocator, file);
    } else {
      status = iree_profile_explain_print_jsonl(
          &summary, &dispatch_context, &memory_context, host_allocator, file);
    }
  }

  iree_profile_memory_context_deinitialize(&memory_context);
  iree_profile_dispatch_context_deinitialize(&dispatch_context);
  iree_profile_summary_deinitialize(&summary);
  iree_profile_file_close(&profile_file);
  return status;
}
