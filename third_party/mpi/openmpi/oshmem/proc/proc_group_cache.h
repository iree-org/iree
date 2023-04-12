/*
 * Copyright (c) 2013-2018 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef _PROC_GROUP_CACHE_H
#define _PROC_GROUP_CACHE_H

#include "oshmem_config.h"
#include "proc.h"

#define OSHMEM_GROUP_CACHE_ENABLED 1

BEGIN_C_DECLS

/**
 * A group cache.
 *
 * Deletion of a group is not implemented because it
 * requires a synchronization between PEs
 *
 * If cache enabled every group is kept until the
 * shmem_finalize() is called
 */

int oshmem_group_cache_init(void);
void oshmem_group_cache_destroy(void);

oshmem_group_t* oshmem_group_cache_find(int pe_start, int pe_stride, int pe_size);

int oshmem_group_cache_insert(oshmem_group_t *group, int pe_start,
                              int pe_stride, int pe_size);

static inline int oshmem_group_cache_enabled(void)
{
    return OSHMEM_GROUP_CACHE_ENABLED;
}

END_C_DECLS

#endif
