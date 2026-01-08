#include "iree/base/allocator_stats.h"

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"

// Size of the prefix we store ahead of the user pointer.
// Must be >= sizeof(iree_host_size_t) and a multiple of iree_max_align_t so
// that the returned fake_ptr is still max-aligned.
#define IREE_ALLOCATOR_STATS_PREFIX_SIZE \
  iree_host_align(sizeof(iree_host_size_t), (iree_host_size_t)iree_max_align_t)

// Records an allocation of |byte_length| bytes in |statistics|.
static inline void iree_allocator_statistics_record_alloc(
    iree_allocator_statistics_t* statistics, iree_host_size_t byte_length) {
  iree_slim_mutex_lock(statistics->mutex);
  statistics->bytes_allocated += byte_length;
  if (statistics->bytes_allocated - statistics->bytes_freed >
      statistics->bytes_peak) {
    statistics->bytes_peak =
        statistics->bytes_allocated - statistics->bytes_freed;
  }
  iree_slim_mutex_unlock(statistics->mutex);
}

// Records a free of |byte_length| bytes in |statistics|.
static inline void iree_allocator_statistics_record_free(
    iree_allocator_statistics_t* statistics, iree_host_size_t byte_length) {
  iree_slim_mutex_lock(statistics->mutex);
  statistics->bytes_freed += byte_length;
  iree_slim_mutex_unlock(statistics->mutex);
}

static inline void* iree_allocator_get_real_alloc_ptr(void* fake_ptr) {
  return (void*)((uint8_t*)fake_ptr - IREE_ALLOCATOR_STATS_PREFIX_SIZE);
}

static inline void* iree_allocator_get_fake_alloc_ptr(void* real_ptr) {
  return (void*)((uint8_t*)real_ptr + IREE_ALLOCATOR_STATS_PREFIX_SIZE);
}

static inline iree_host_size_t iree_allocator_get_alloc_size(void* real_ptr) {
  return *(iree_host_size_t*)real_ptr;
}

static inline void iree_allocator_set_alloc_size(void* real_ptr,
                                                 iree_host_size_t byte_length) {
  *(iree_host_size_t*)real_ptr = byte_length;
}

IREE_API_EXPORT iree_status_t
iree_allocator_stats_ctl(void* self, iree_allocator_command_t command,
                         const void* params, void** inout_ptr) {
  iree_allocator_with_stats_t* stats_allocator =
      (iree_allocator_with_stats_t*)self;
  void* base_self = stats_allocator->base_allocator.self;
  iree_allocator_statistics_t* statistics = &stats_allocator->statistics;
  iree_allocator_ctl_fn_t base_ctl = stats_allocator->base_allocator.ctl;
  if (IREE_UNLIKELY(!base_ctl)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator has no control routine");
  }

  if (command == IREE_ALLOCATOR_COMMAND_FREE) {
    if (!*inout_ptr) {
      return iree_ok_status();
    }
    void* real_ptr = iree_allocator_get_real_alloc_ptr(*inout_ptr);
    iree_host_size_t byte_length = iree_allocator_get_alloc_size(real_ptr);
    iree_status_t status =
        base_ctl(base_self, command, /*params=*/NULL, &real_ptr);
    if (iree_status_is_ok(status)) {
      iree_allocator_statistics_record_free(statistics, byte_length);
    }
    return status;
  }

  // Prepare adjusted allocation params to account for stored size, we need to
  // store the size of the allocation to be able to correctly account for the
  // memory usage of reallocs and frees
  iree_host_size_t byte_length =
      ((const iree_allocator_alloc_params_t*)params)->byte_length;
  iree_allocator_alloc_params_t new_params = {
      .byte_length = byte_length + IREE_ALLOCATOR_STATS_PREFIX_SIZE,
  };

  if (command == IREE_ALLOCATOR_COMMAND_REALLOC && *inout_ptr) {
    void* real_ptr = iree_allocator_get_real_alloc_ptr(*inout_ptr);
    iree_host_size_t old_byte_length = iree_allocator_get_alloc_size(real_ptr);
    iree_status_t status = base_ctl(base_self, command, &new_params, &real_ptr);
    if (iree_status_is_ok(status)) {
      iree_allocator_set_alloc_size(real_ptr, byte_length);

      // Check if the pointer changed if so we treat this as a free + alloc,
      // otherwise it's either growing or shrinking the existing allocation
      if (real_ptr != iree_allocator_get_real_alloc_ptr(*inout_ptr)) {
        iree_allocator_statistics_record_free(statistics, old_byte_length);
        iree_allocator_statistics_record_alloc(statistics, byte_length);
      } else if (byte_length > old_byte_length) {
        iree_allocator_statistics_record_alloc(statistics,
                                               byte_length - old_byte_length);
      } else if (byte_length < old_byte_length) {
        iree_allocator_statistics_record_free(statistics,
                                              old_byte_length - byte_length);
      }

      *inout_ptr = iree_allocator_get_fake_alloc_ptr(real_ptr);
    } else {
      // Forward any pointer value provided by the allocator on failure.
      *inout_ptr = real_ptr;
    }
    return status;
  }

  // New allocation.
  void* real_ptr = NULL;
  iree_status_t status = base_ctl(base_self, command, &new_params, &real_ptr);
  if (iree_status_is_ok(status)) {
    iree_allocator_set_alloc_size(real_ptr, byte_length);
    iree_allocator_statistics_record_alloc(statistics, byte_length);
    *inout_ptr = iree_allocator_get_fake_alloc_ptr(real_ptr);
  } else {
    // Forward any pointer value provided by the allocator on failure.
    *inout_ptr = real_ptr;
  }
  return status;
}

IREE_API_EXPORT iree_allocator_t
iree_allocator_stats_init(iree_allocator_with_stats_t* stats_allocator,
                          iree_allocator_t base_allocator) {
  stats_allocator->base_allocator = base_allocator;
  memset(&stats_allocator->statistics, 0, sizeof(iree_allocator_statistics_t));
  iree_allocator_malloc(base_allocator, sizeof(iree_slim_mutex_t),
                        (void**)&stats_allocator->statistics.mutex);
  iree_slim_mutex_initialize(stats_allocator->statistics.mutex);
  iree_allocator_t v = {
      .ctl = iree_allocator_stats_ctl,
      .self = stats_allocator,
  };
  return v;
}

IREE_API_EXPORT void iree_allocator_stats_deinit(
    iree_allocator_with_stats_t* stats_allocator) {
  iree_slim_mutex_deinitialize(stats_allocator->statistics.mutex);
  iree_allocator_free(stats_allocator->base_allocator,
                      stats_allocator->statistics.mutex);
}

IREE_API_EXPORT iree_status_t iree_allocator_statistics_fprint(
    FILE* file, iree_allocator_with_stats_t* allocator) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(allocator->base_allocator, &builder);
  iree_status_t status = iree_string_builder_append_cstring(
      &builder, "[[ iree_allocator_t memory statistics ]]\n");

  iree_allocator_statistics_t* stats = &allocator->statistics;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(stats->mutex);
    status = iree_string_builder_append_format(
        &builder,
        "  HOST_ALLOC: %12" PRIdsz "B peak / %12" PRIdsz
        "B allocated / %12" PRIdsz "B freed / %12" PRIdsz "B live\n",
        stats->bytes_peak, stats->bytes_allocated, stats->bytes_freed,
        (stats->bytes_allocated - stats->bytes_freed));
    iree_slim_mutex_unlock(stats->mutex);
  }

  if (iree_status_is_ok(status)) {
    fprintf(file, "%.*s", (int)iree_string_builder_size(&builder),
            iree_string_builder_buffer(&builder));
  }

  iree_string_builder_deinitialize(&builder);
  return status;
}
