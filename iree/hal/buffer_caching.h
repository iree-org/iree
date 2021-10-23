#ifndef IREE_HAL_BUFFER_CACHING_H_
#define IREE_HAL_BUFFER_CACHING_H_

#include "iree/hal/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_buffer_cache {
    iree_hal_buffer_t *buffer;
    struct iree_hal_buffer_cache *next;
} iree_hal_buffer_cache;

extern iree_hal_buffer_cache *iree_hal_buffer_cache_list;

iree_status_t iree_hal_add_buffer_to_cache(iree_hal_buffer_t *buffer);

bool iree_hal_cached_buffer_available(iree_host_size_t requested_size);

void iree_hal_remove_buffer_from_cache(int buffer_index_to_remove);

iree_status_t iree_hal_allocate_cached_buffer(iree_host_size_t requested_size, iree_hal_buffer_t** out_buffer);

void iree_hal_clear_buffer(const char* driver_name);

void iree_hal_clear_cuda_buffer();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_BUFFER_CACHING_H_