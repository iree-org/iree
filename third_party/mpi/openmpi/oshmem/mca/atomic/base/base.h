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

#ifndef MCA_ATOMIC_BASE_H
#define MCA_ATOMIC_BASE_H

#include "oshmem_config.h"

#include "oshmem/mca/atomic/atomic.h"
#include "opal/class/opal_list.h"

/*
 * Global functions for MCA overall atomic open and close
 */

BEGIN_C_DECLS

int mca_atomic_base_find_available(bool enable_progress_threads,
                                   bool enable_threads);

int mca_atomic_base_select(void);

/*
 * MCA framework
 */
OSHMEM_DECLSPEC extern mca_base_framework_t oshmem_atomic_base_framework;

/* ******************************************************************** */
#ifdef __BASE_FILE__
#define __ATOMIC_FILE__ __BASE_FILE__
#else
#define __ATOMIC_FILE__ __FILE__
#endif

#ifdef OPAL_ENABLE_DEBUG
#define ATOMIC_VERBOSE(level, ...) \
    oshmem_output_verbose(level, oshmem_atomic_base_framework.framework_output, \
        "%s:%d - %s()", __ATOMIC_FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define ATOMIC_VERBOSE(level, ...)
#endif

#define ATOMIC_ERROR(...) \
    oshmem_output(oshmem_atomic_base_framework.framework_output, \
        "Error %s:%d - %s()", __ATOMIC_FILE__, __LINE__, __func__, __VA_ARGS__)

END_C_DECLS

#endif /* MCA_ATOMIC_BASE_H */
