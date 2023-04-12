/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_SCOLL_BASE_H
#define MCA_SCOLL_BASE_H

#include "oshmem_config.h"

#include "oshmem/mca/memheap/memheap.h"
#include "opal/class/opal_list.h"

/*
 * Global functions for MCA overall collective open and close
 */

BEGIN_C_DECLS

/*
 * Globals
 */
/**
 * Special synchronization array to do barrier all.
 */
OSHMEM_DECLSPEC extern long* mca_scoll_sync_array;

OSHMEM_DECLSPEC int mca_scoll_base_find_available(bool enable_progress_threads,
                                                  bool enable_threads);

OSHMEM_DECLSPEC int mca_scoll_base_select(struct oshmem_group_t *group);

int mca_scoll_base_group_unselect(struct oshmem_group_t *group);

OSHMEM_DECLSPEC int mca_scoll_enable(void);

/*
 * MCA framework
 */
OSHMEM_DECLSPEC extern mca_base_framework_t oshmem_scoll_base_framework;

/* ******************************************************************** */
#ifdef __BASE_FILE__
#define __SCOLL_FILE__ __BASE_FILE__
#else
#define __SCOLL_FILE__ __FILE__
#endif

#ifdef OPAL_ENABLE_DEBUG
#define SCOLL_VERBOSE(level, ...) \
    oshmem_output_verbose(level, oshmem_scoll_base_framework.framework_output, \
       "%s:%d - %s()", __SCOLL_FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SCOLL_VERBOSE(...)
#endif

#define SCOLL_ERROR(...) \
    oshmem_output(oshmem_scoll_base_framework.framework_output, \
        "Error %s:%d - %s()",  __SCOLL_FILE__, __LINE__, __func__, __VA_ARGS__)

END_C_DECLS

#endif /* MCA_SCOLL_BASE_H */
