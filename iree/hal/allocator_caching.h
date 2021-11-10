#ifndef IREE_HAL_ALLOCATOR_CACHING_H_
#define IREE_HAL_ALLOCATOR_CACHING_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_caching_allocator_t
//===----------------------------------------------------------------------===//

// Creates an allocator which will cache buffers, fulfilling any cache misses with the
// delegate_allocator.
IREE_API_EXPORT iree_status_t iree_hal_allocator_create_caching(
    iree_string_view_t identifier, iree_hal_allocator_t* delegate_allocator,
    iree_hal_allocator_t** out_allocator);

IREE_API_EXPORT iree_status_t iree_hal_allocator_add_buffer_to_cache(
    iree_hal_buffer_t* base_buffer);

// Boolean value to denote if the buffers should be cached or destroyed.
IREE_API_EXPORT extern bool iree_hal_allocator_cache_buffer;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ALLOCATOR_CACHING_H_
