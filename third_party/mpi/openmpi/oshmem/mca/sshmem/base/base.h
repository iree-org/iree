/*
 * Copyright (c) 2014      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_SSHMEM_BASE_H
#define MCA_SSHMEM_BASE_H

#include "oshmem_config.h"
#include "oshmem/mca/sshmem/sshmem.h"
#include "oshmem/proc/proc.h"

#include "opal/mca/base/mca_base_framework.h"

#include "orte/runtime/orte_globals.h"

BEGIN_C_DECLS

extern void* mca_sshmem_base_start_address;
extern char* mca_sshmem_base_backing_file_dir;

/* ////////////////////////////////////////////////////////////////////////// */
/* Public API for the sshmem framework */
/* ////////////////////////////////////////////////////////////////////////// */
OSHMEM_DECLSPEC int
mca_sshmem_segment_create(map_segment_t *ds_buf,
                          const char *file_name,
                          size_t size, long hint);

OSHMEM_DECLSPEC void *
mca_sshmem_segment_attach(map_segment_t *ds_buf, sshmem_mkey_t *mkey);

OSHMEM_DECLSPEC int
mca_sshmem_segment_detach(map_segment_t *ds_buf, sshmem_mkey_t *mkey);

OSHMEM_DECLSPEC int
mca_sshmem_unlink(map_segment_t *ds_buf);
/* ////////////////////////////////////////////////////////////////////////// */
/* End Public API for the sshmem framework */
/* ////////////////////////////////////////////////////////////////////////// */

/*
 * Global functions for MCA overall sshmem open and close
 */

/**
 * Select an available component.
 *
 * @return OSHMEM_SUCCESS Upon success.
 * @return OSHMEM_NOT_FOUND If no component can be selected.
 * @return OSHMEM_ERROR Upon other failure.
 *
 * This function invokes the selection process for shmem components,
 * which works as follows:
 *
 * - If the \em sshmem MCA parameter is not specified, the
 *   selection set is all available shmem components.
 * - If the \em sshmem MCA parameter is specified, the
 *   selection set is just that component.
 * - All components in the selection set are queried to see if
 *   they want to run.  All components that want to run are ranked
 *   by their priority and the highest priority component is
 *   selected.  All non-selected components have their "close"
 *   function invoked to let them know that they were not selected.
 * - The selected component will have its "init" function invoked to
 *   let it know that it was selected.
 *
 * If we fall through this entire process and no component is
 * selected, then return OSHMEM_NOT_FOUND (this is not a fatal
 * error).
 *
 * At the end of this process, we'll either have a single
 * component that is selected and initialized, or no component was
 * selected.  If no component was selected, subsequent invocation
 * of the shmem wrapper functions will return an error.
 */
OSHMEM_DECLSPEC int
mca_sshmem_base_select(void);

/**
 * Shut down the sshmem MCA framework.
 *
 * @retval OSHMEM_SUCCESS Always
 *
 * This function shuts down everything in the sshmem MCA
 * framework, and is called during opal_finalize().
 *
 * It must be the last function invoked on the sshmem MCA
 * framework.
 */
OSHMEM_DECLSPEC int
mca_sshmem_base_close(void);

/**
 * Indication of whether a component was successfully selected or
 * not
 */
OSHMEM_DECLSPEC extern bool mca_sshmem_base_selected;

/**
 * Global component struct for the selected component
 */
OSHMEM_DECLSPEC extern const mca_sshmem_base_component_2_0_0_t
*mca_sshmem_base_component;

/**
 * Global module struct for the selected module
 */
OSHMEM_DECLSPEC extern const mca_sshmem_base_module_2_0_0_t
*mca_sshmem_base_module;

/**
 * Framework structure declaration
 */
OSHMEM_DECLSPEC extern mca_base_framework_t oshmem_sshmem_base_framework;

/* ******************************************************************** */
#ifdef __BASE_FILE__
#define __SSHMEM_FILE__ __BASE_FILE__
#else
#define __SSHMEM_FILE__ __FILE__
#endif

#if OPAL_ENABLE_DEBUG
#define SSHMEM_VERBOSE(level, ...) \
    oshmem_output_verbose(level, oshmem_sshmem_base_framework.framework_output, \
        "%s:%d - %s()", __SSHMEM_FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SSHMEM_VERBOSE(level, ...)
#endif

#define SSHMEM_ERROR(...) \
    oshmem_output(oshmem_sshmem_base_framework.framework_output, \
        "Error %s:%d - %s()", __SSHMEM_FILE__, __LINE__, __func__, __VA_ARGS__)

#define SSHMEM_WARN(...) \
    oshmem_output_verbose(0, oshmem_sshmem_base_framework.framework_output, \
        "Warning %s:%d - %s()", __SSHMEM_FILE__, __LINE__, __func__, __VA_ARGS__)


OSHMEM_DECLSPEC extern void shmem_ds_reset(map_segment_t *ds_buf);

/*
 * Get unique file name
 */
OSHMEM_DECLSPEC extern char * oshmem_get_unique_file_name(uint64_t pe);

END_C_DECLS

#endif /* MCA_SSHMEM_BASE_H */
