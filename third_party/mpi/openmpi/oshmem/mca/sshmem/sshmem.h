/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2014      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * sshmem (shared memory backing facility) framework component interface
 * definitions.
 *
 * The module has the following functions:
 *
 * - module_init
 * - segment_create
 * - segment_attach
 * - segment_detach
 * - unlink
 * - module_finalize
 */

#ifndef MCA_SSHMEM_H
#define MCA_SSHMEM_H

#include "oshmem_config.h"
#include "oshmem/types.h"
#include "oshmem/constants.h"

#include "oshmem/mca/mca.h"
#include "opal/mca/base/base.h"

#include "oshmem/mca/sshmem/sshmem_types.h"

BEGIN_C_DECLS

/* ////////////////////////////////////////////////////////////////////////// */
typedef int
(*mca_sshmem_base_component_runtime_query_fn_t)(mca_base_module_t **module,
                                                int *priority,
                                                const char *hint);

/* structure for sshmem components. */
struct mca_sshmem_base_component_2_0_0_t {
    /* base MCA component */
    mca_base_component_t base_version;
    /* base MCA data */
    mca_base_component_data_t base_data;
    /* component runtime query */
    mca_sshmem_base_component_runtime_query_fn_t runtime_query;
};

/* convenience typedefs */
typedef struct mca_sshmem_base_component_2_0_0_t
mca_sshmem_base_component_2_0_0_t;

typedef struct mca_sshmem_base_component_2_0_0_t mca_sshmem_base_component_t;

/* ////////////////////////////////////////////////////////////////////////// */
/* shmem API function pointers */

/**
 * module initialization function.
 * @return OSHMEM_SUCCESS on success.
 */
typedef int
(*mca_sshmem_base_module_init_fn_t)(void);

/**
 * create a new shared memory segment and initialize members in structure
 * pointed to by ds_buf.
 *
 * @param ds_buf               pointer to map_segment_t typedef'd structure
 *                             defined in shmem_types.h (OUT).
 *
 * @param file_name file_name  unique string identifier that must be a valid,
 *                             writable path (IN).
 *
 * @param address              address to attach the segment at, or 0 allocate
 *                             any available address in the process.
 *
 * @param size                 size of the shared memory segment.
 *
 * @param hint                 hint of the shared memory segment.
 *
 * @return OSHMEM_SUCCESS on success.
 */
typedef int
(*mca_sshmem_base_module_segment_create_fn_t)(map_segment_t *ds_buf,
                                              const char *file_name,
                                              size_t size, long hint);

/**
 * attach to an existing shared memory segment initialized by segment_create.
 *
 * @param ds_buf  pointer to initialized map_segment_t typedef'd
 *                structure (IN/OUT).
 *
 * @return        base address of shared memory segment on success. returns
 *                NULL otherwise.
 */
typedef void *
(*mca_sshmem_base_module_segment_attach_fn_t)(map_segment_t *ds_buf, sshmem_mkey_t *mkey);

/**
 * detach from an existing shared memory segment.
 *
 * @param ds_buf  pointer to initialized map_segment_t typedef'd structure
 *                (IN/OUT).
 *
 * @return OSHMEM_SUCCESS on success.
 */
typedef int
(*mca_sshmem_base_module_segment_detach_fn_t)(map_segment_t *ds_buf, sshmem_mkey_t *mkey);

/**
 * unlink an existing shared memory segment.
 *
 * @param ds_buf  pointer to initialized map_segment_t typedef'd structure
 *                (IN/OUT).
 *
 * @return OSHMEM_SUCCESS on success.
 */
typedef int
(*mca_sshmem_base_module_unlink_fn_t)(map_segment_t *ds_buf);

/**
 * module finalize function.  invoked by the base on the selected
 * module when the sshmem framework is being shut down.
 */
typedef int (*mca_sshmem_base_module_finalize_fn_t)(void);

/**
 * structure for shmem modules
 */
struct mca_sshmem_base_module_2_0_0_t {
    mca_sshmem_base_module_init_fn_t            module_init;
    mca_sshmem_base_module_segment_create_fn_t  segment_create;
    mca_sshmem_base_module_segment_attach_fn_t  segment_attach;
    mca_sshmem_base_module_segment_detach_fn_t  segment_detach;
    mca_sshmem_base_module_unlink_fn_t          unlink;
    mca_sshmem_base_module_finalize_fn_t        module_finalize;
};

/**
 * convenience typedefs
 */
typedef struct mca_sshmem_base_module_2_0_0_t mca_sshmem_base_module_2_0_0_t;
typedef struct mca_sshmem_base_module_2_0_0_t mca_sshmem_base_module_t;

/**
 * macro for use in components that are of type sshmem
 * see: oshmem/mca/mca.h for more information
 */
#define MCA_SSHMEM_BASE_VERSION_2_0_0                                          \
    OSHMEM_MCA_BASE_VERSION_2_1_0("sshmem", 2, 0, 0)

END_C_DECLS

#endif /* MCA_SSHMEM_H */
