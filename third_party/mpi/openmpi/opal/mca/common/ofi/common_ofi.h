/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2015      Intel, Inc. All rights reserved.
 * Copyright (c) 2017      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2020      Triad National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2021      Amazon.com, Inc. or its affiliates. All rights
 *                         reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_MCA_COMMON_OFI_H
#define OPAL_MCA_COMMON_OFI_H

#include "opal/util/proc.h"
#include "opal/memoryhooks/memory.h"

BEGIN_C_DECLS

typedef struct opal_common_ofi_module {
    char **prov_include;
    char **prov_exclude;
    int output;
} opal_common_ofi_module_t;

extern opal_common_ofi_module_t opal_common_ofi;

/**
 * Common MCA registration
 *
 * Common MCA registration handlinge.  After calling this function,
 * \code opal_common_ofi.output will be properly initialized.
 *
 * @param component (IN) OFI component being initialized
 *
 * @returns OPAL_SUCCESS on success, OPAL error code on failure
 */
OPAL_DECLSPEC int opal_common_ofi_mca_register(const mca_base_component_t *component);

/**
 * Initializes common objects for libfabric
 *
 * Initialize common libfabric interface.  This should be called from
 * any other OFI component's component_open() call.
 *
 * @note This function is not thread safe and must be called in a
 * serial portion of the code.
 */
OPAL_DECLSPEC int opal_common_ofi_open(void);

/**
 * Cleans up common objects for libfabric
 *
 * Clean up common libfabric interface.  This should be called from
 * any other OFI component's component_close() call.  Resource cleanup
 * is reference counted, so any successful call to
 * opal_common_ofi_init().
 *
 * @note This function is not thread safe and must be called in a
 * serial portion of the code.
 */
OPAL_DECLSPEC int opal_common_ofi_close(void);

/**
 * Export our memory hooks into Libfabric monitor
 *
 * Use Open MPI's memory hooks to provide monitor notifications to
 * Libfabric via the external mr_cache facility.  This must be called
 * before any domain is initialized (ie, before any Libfabric memory
 * monitor is configured).
 *
 * @returns A libfabric error code is returned on error
 */
OPAL_DECLSPEC int opal_common_ofi_export_memory_monitor(void);

/**
 * Search function for provider names
 *
 * This function will take a provider name string and a list of lower
 * provider name strings as inputs. It will return true if the lower
 * provider in the item string matches a lower provider in the list.
 *
 * @param list (IN)    List of strings corresponding to lower providers.
 * @param item (IN)    Single string corresponding to a provider.
 *
 * @return 0           The lower provider of the item string is not in
 *                     list or an input was NULL
 * @return 1           The lower provider of the item string matches
 *                     a string in the item list.
 *
 */
OPAL_DECLSPEC int opal_common_ofi_is_in_list(char **list, char *item);

/**
 * Selects NIC (provider) based on hardware locality
 *
 * There are 3 main cases that this covers:
 *
 *      1. If the first provider passed into this function is the only valid
 *      provider, this provider is returned.
 *
 *      2. If there is more than 1 provider that matches the type of the first
 *      provider in the list, and the BDF data
 *      is available then a provider is selected based on locality of device
 *      cpuset and process cpuset and tries to ensure that processes are distributed
 *      evenly across NICs. This has two separate cases:
 *
 *          i. There is one or more provider local to the process:
 *
 *              (local rank % number of providers of the same type that share the process cpuset)
 *              is used to select one of these providers.
 *
 *          ii. There is no provider that is local to the process:
 *
 *              (local rank % number of providers of the same type)
 *              is used to select one of these providers
 *
 *      3. If there is more than 1 providers of the same type in the list, and the BDF data
 *      is not available (the ofi version does not support fi_info.nic or the
 *      provider does not support BDF) then (local rank % number of providers of the same type)
 *      is used to select one of these providers
 *
 *      @param provider_list (IN)   struct fi_info* An initially selected
 *                                  provider NIC. The provider name and
 *                                  attributes are used to restrict NIC
 *                                  selection. This provider is returned if the
 *                                  NIC selection fails.
 *
 *      @param package_rank (IN)   uint32_t The rank of the process. Used to
 *                                  select one valid NIC if there is a case
 *                                  where more than one can be selected. This
 *                                  could occur when more than one provider
 *                                  shares the same cpuset as the process.
 *                                  This could either be a package_rank if one is
 *                                  successfully calculated, or the process id.
 *
 *      @param provider (OUT)       struct fi_info* object with the selected
 *                                  provider if the selection succeeds
 *                                  if the selection fails, returns the fi_info
 *                                  object that was initially provided.
 *
 * All errors should be recoverable and will return the initially provided
 * provider. However, if an error occurs we can no longer guarantee
 * that the provider returned is local to the process or that the processes will
 * balance across available NICs.
 */
OPAL_DECLSPEC struct fi_info* opal_mca_common_ofi_select_provider(struct fi_info *provider_list,
                                                                  int32_t num_local_peers,
                                                                  uint16_t my_local_rank,
                                                                  char *cpuset, uint32_t pid);

END_C_DECLS

#endif /* OPAL_MCA_COMMON_OFI_H */
