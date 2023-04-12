/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2016      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_SPML_BASE_H
#define MCA_SPML_BASE_H

#include "oshmem_config.h"

#include "oshmem/mca/mca.h"
#include "opal/mca/base/mca_base_framework.h"
#include "opal/class/opal_list.h"
#include "opal/class/opal_pointer_array.h"

#include "oshmem/mca/spml/spml.h"

/*
 * Global functions for the PML
 */

BEGIN_C_DECLS

/*
 * This is the base priority for a SPML wrapper component
 * If there exists more than one then it is undefined
 * which one is picked.
 */
#define SPML_SELECT_WRAPPER_PRIORITY -128

/*
 * Globals
 */
OSHMEM_DECLSPEC extern mca_spml_base_component_t mca_spml_base_selected_component;
OSHMEM_DECLSPEC extern opal_pointer_array_t mca_spml_base_spml;

OSHMEM_DECLSPEC int mca_spml_base_finalize(void);

/*
 * Select an available component.
 */
OSHMEM_DECLSPEC  int mca_spml_base_select(bool enable_progress_threads,
                                          bool enable_threads);

/*
 * Share in modex the name of the selected component
 */
OSHMEM_DECLSPEC int mca_spml_base_spml_selected(const char *name);

/*
 * Verify that all new procs are using the currently selected component
 */
OSHMEM_DECLSPEC int mca_spml_base_spml_check_selected(const char *my_spml,
                                                      ompi_proc_t **procs,
                                                      size_t nprocs);

OSHMEM_DECLSPEC int mca_spml_base_wait(void* addr,
                                       int cmp,
                                       void* value,
                                       int datatype);
OSHMEM_DECLSPEC int mca_spml_base_wait_nb(void* handle);
OSHMEM_DECLSPEC int mca_spml_base_test(void* addr,
                                       int cmp,
                                       void* value,
                                       int datatype,
                                       int *out_value);
OSHMEM_DECLSPEC int mca_spml_base_oob_get_mkeys(shmem_ctx_t ctx,
                                                int pe,
                                                uint32_t seg,
                                                sshmem_mkey_t *mkeys);

OSHMEM_DECLSPEC void mca_spml_base_rmkey_unpack(shmem_ctx_t ctx, sshmem_mkey_t *mkey, uint32_t seg, int pe, int tr_id);
OSHMEM_DECLSPEC void mca_spml_base_rmkey_free(sshmem_mkey_t *mkey, int pe);
OSHMEM_DECLSPEC void *mca_spml_base_rmkey_ptr(const void *dst_addr, sshmem_mkey_t *mkey, int pe);

OSHMEM_DECLSPEC int mca_spml_base_put_nb(void *dst_addr,
                                         size_t size,
                                         void *src_addr,
                                         int dst,
                                         void **handle);
OSHMEM_DECLSPEC int mca_spml_base_get_nb(void *dst_addr,
                                         size_t size,
                                         void *src_addr,
                                         int src,
                                         void **handle);

OSHMEM_DECLSPEC void mca_spml_base_memuse_hook(void *addr, size_t length);

OSHMEM_DECLSPEC int mca_spml_base_put_all_nb(void *target, const void *source,
                                             size_t size, long *counter);

/*
 * MCA framework
 */
OSHMEM_DECLSPEC extern mca_base_framework_t oshmem_spml_base_framework;

/* ******************************************************************** */
#ifdef __BASE_FILE__
#define __SPML_FILE__ __BASE_FILE__
#else
#define __SPML_FILE__ __FILE__
#endif

#ifdef OPAL_ENABLE_DEBUG
#define SPML_VERBOSE(level, ...) \
    oshmem_output_verbose(level, oshmem_spml_base_framework.framework_output, \
        "%s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SPML_VERBOSE(level, ...)
#endif

#define SPML_VERBOSE_FASTPATH(level, ...)

#define SPML_ERROR(...) \
    oshmem_output(oshmem_spml_base_framework.framework_output, \
        "Error %s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)

#define SPML_WARNING(...) \
    oshmem_output_verbose(0, oshmem_spml_base_framework.framework_output, \
        "Warning %s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)

END_C_DECLS

#endif /* MCA_SPML_BASE_H */
