#ifndef EXPERIMENTAL_HAL_ALLOCATOR_CACHING_H_
#define EXPERIMENTAL_HAL_ALLOCATOR_CACHING_H_

#include "iree/base/api.h"
#include "iree/hal/allocator.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_caching_allocator_t
//===----------------------------------------------------------------------===//

// Creates an allocator which will cache buffers, fulfilling any cache misses
// with the delegate_allocator.
IREE_API_EXPORT iree_status_t iree_hal_allocator_create_caching(
    iree_hal_allocator_t* delegate_allocator,
    iree_hal_allocator_t** out_allocator);

IREE_API_EXPORT iree_status_t
iree_hal_allocator_add_buffer_to_cache(iree_hal_buffer_t* base_buffer);

IREE_API_EXPORT iree_hal_allocator_t*
iree_hal_caching_allocator_get_delegate(iree_hal_allocator_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_HAL_ALLOCATOR_CACHING_H_
