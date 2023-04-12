/*
 *  Copyright (c) 2013      Mellanox Technologies, Inc.
 *                          All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#if !defined(OSHMEM_INFO_SUPPORT_H)
#define OSHMEM_INFO_SUPPORT_H

#include "oshmem_config.h"

#include "opal/class/opal_pointer_array.h"

OSHMEM_DECLSPEC void oshmem_info_register_types(opal_pointer_array_t *mca_types);

OSHMEM_DECLSPEC int oshmem_info_register_framework_params(opal_pointer_array_t *component_map);

OSHMEM_DECLSPEC void oshmem_info_close_components(void);

OSHMEM_DECLSPEC void oshmem_info_show_oshmem_version(const char *scope);

#endif /* !defined(OSHMEM_INFO_SUPPORT_H) */
