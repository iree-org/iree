#include "iree/hal/buffer_caching.h"

iree_hal_buffer_cache *iree_hal_buffer_cache_list = NULL;

iree_status_t iree_hal_add_buffer_to_cache(iree_hal_buffer_t* buffer) {
    struct iree_hal_buffer_cache* iree_hal_buffer_node = 
                        (iree_hal_buffer_cache*) malloc(sizeof(iree_hal_buffer_cache));
    iree_hal_buffer_node->buffer = buffer;
    iree_hal_buffer_node->next = NULL;
    if(iree_hal_buffer_cache_list == NULL) {
        iree_hal_buffer_cache_list = iree_hal_buffer_node;
    } else {
        iree_hal_buffer_cache *iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_list;
        // Add buffer to the end of list
        while(iree_hal_buffer_cache_ptr != NULL) {
            if(iree_hal_buffer_cache_ptr->next == NULL)
                break;
            else
                iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_ptr->next;
        }
        iree_hal_buffer_cache_ptr->next = iree_hal_buffer_node;
    }
    return iree_ok_status();
}

bool iree_hal_cached_buffer_available(iree_host_size_t requested_size) {
    iree_hal_buffer_cache *iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_list;
    while(iree_hal_buffer_cache_ptr != NULL) {
        if(iree_hal_buffer_cache_ptr->buffer->allocation_size >= requested_size) {
            return true;
        }
        iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_ptr->next;
    }
    return false;
}

void iree_hal_remove_buffer_from_cache(int buffer_index_to_remove) {

    iree_hal_buffer_cache *buffer_to_be_removed_from_cache, *iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_list;
    if(buffer_index_to_remove == 0) {
        buffer_to_be_removed_from_cache = iree_hal_buffer_cache_ptr;
        if(buffer_to_be_removed_from_cache->next == NULL) {
            iree_hal_buffer_cache_list = NULL;
        } else {
            iree_hal_buffer_cache_list = iree_hal_buffer_cache_list->next;
        }
    } else {
        int current_buffer_index = 0;
        while(current_buffer_index < buffer_index_to_remove-1) {
            iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_ptr->next;
            current_buffer_index++;
        }
        buffer_to_be_removed_from_cache = iree_hal_buffer_cache_ptr->next;

        if(buffer_to_be_removed_from_cache->next == NULL) {
            iree_hal_buffer_cache_ptr->next = NULL;
        } else {
            iree_hal_buffer_cache_ptr->next = buffer_to_be_removed_from_cache->next;
        }
    }
    free(buffer_to_be_removed_from_cache);
    return;
}

iree_status_t iree_hal_allocate_cached_buffer(iree_host_size_t requested_size, iree_hal_buffer_t** out_buffer) {
    size_t buffer_index_in_cache = 0;
    iree_hal_buffer_cache *iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_list;
    while(iree_hal_buffer_cache_ptr != NULL) {
        if(iree_hal_buffer_cache_ptr->buffer->allocation_size >= requested_size) {
            break;
        }
        iree_hal_buffer_cache_ptr = iree_hal_buffer_cache_ptr->next;
        buffer_index_in_cache++;
    }
    iree_hal_buffer_t *buffer_to_allocate = iree_hal_buffer_cache_ptr->buffer;
    
    *out_buffer = buffer_to_allocate;
    iree_hal_remove_buffer_from_cache(buffer_index_in_cache);

    return iree_ok_status();
}

void iree_hal_clear_buffer(const char* driver_name) {
    char cuda_name_string[] = "cuda"; 
    if(!strcmp(driver_name, cuda_name_string)) {
        iree_hal_clear_cuda_buffer();
    }
    return;
}