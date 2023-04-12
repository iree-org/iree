/*
 * Copyright (c) 2009-2012 Mellanox Technologies.  All rights reserved.
 *                         All rights reserved.
 * Copyright (c) 2009-2012 Oak Ridge National Laboratory.  All rights reserved.
 * Copyright (c) 2012-2015 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2014      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef _COMMON_OFAUTILS_H_
#define _COMMON_OFAUTILS_H_

#include "opal_config.h"

#include <stdint.h>
#include <infiniband/verbs.h>

#include "opal/mca/mca.h"

#include <infiniband/verbs.h>

#include "opal/class/opal_list.h"

BEGIN_C_DECLS

/*
 * common_verbs_devlist.c
 */
OPAL_DECLSPEC struct ibv_device **opal_ibv_get_device_list(int *num_devs);
OPAL_DECLSPEC void opal_ibv_free_device_list(struct ibv_device **ib_devs);

/*
 * common_verbs_mca.c
 */
extern bool opal_common_verbs_warn_nonexistent_if;
extern int opal_common_verbs_want_fork_support;
OPAL_DECLSPEC void opal_common_verbs_mca_register(mca_base_component_t *component);

/*
 * common_verbs_basics.c
 */
bool opal_common_verbs_check_basics(void);

/*
 * common_verbs_find_ports.c
 */
typedef struct opal_common_verbs_device_item_t {
    opal_object_t super;

    struct ibv_device *device;
    char *device_name;
    struct ibv_context *context;
    struct ibv_device_attr device_attr;

    /** This field defaults to true, meaning that the destructor for
        opal_common_verbs_device_item_t will invoke ibv_close_device()
        on the context.  An upper layer can reset this field to false,
        however, indicating that the destructor should *not* invoke
        ibv_close_device() (e.g., if the upper layer has copied the
        context and is using it). */
    bool destructor_free_context;
} opal_common_verbs_device_item_t;
OBJ_CLASS_DECLARATION(opal_common_verbs_device_item_t);

typedef struct opal_common_verbs_port_item_t {
    opal_list_item_t super;

    opal_common_verbs_device_item_t *device;
    uint8_t port_num;
    struct ibv_port_attr port_attr;
} opal_common_verbs_port_item_t;
OBJ_CLASS_DECLARATION(opal_common_verbs_port_item_t);

enum {
    OPAL_COMMON_VERBS_FLAGS_RC = 0x1,
    OPAL_COMMON_VERBS_FLAGS_NOT_RC = 0x2,
    OPAL_COMMON_VERBS_FLAGS_UD = 0x4,
    OPAL_COMMON_VERBS_FLAGS_TRANSPORT_IB = 0x8,
    OPAL_COMMON_VERBS_FLAGS_TRANSPORT_IWARP = 0x10,
    /* Note that these 2 link layer flags will only be useful if
       defined(HAVE_IBV_LINK_LAYER_ETHERNET). Otherwise, they will be
       ignored. */
    OPAL_COMMON_VERBS_FLAGS_LINK_LAYER_IB = 0x80,
    OPAL_COMMON_VERBS_FLAGS_LINK_LAYER_ETHERNET = 0x100,
    OPAL_COMMON_VERBS_FLAGS_MAX
};

/**
 * Find a list of ibv_device ports that match a specific criteria.
 *
 * @param if_include (IN): comma-delimited list of interfaces to use
 * @param if_exclude (IN): comma-delimited list of interfaces to NOT use
 * @param flags (IN): bit flags
 * @param verbose_stream (IN): stream to send opal_output_verbose messages to
 *
 * The ports will adhere to the if_include / if_exclude lists (only
 * one can be specified).  The lists are comma-delimited tokens in one
 * of two forms:
 *
 * interface_name
 * interface_name:port
 *
 * Hence, a if_include list could be the follwing: "mlx4_0,mthca0:1".
 *
 * The flags provide logical OR behavior -- a port will be included if
 * it includes any of the capabilities/characteristics listed in the
 * flags.
 *
 * Note that if the verbose_stream is >=0, output will be sent to that
 * stream with a verbose level of 5.
 *
 * A valid list will always be returned.  It will contain zero or more
 * opal_common_verbs_port_item_t items.  Each item can be individually
 * OBJ_RELEASE'd; the destructor will take care of cleaning up the
 * linked opal_common_verbs_device_item_t properly (i.e., when all
 * port_items referring to it have been freed).
 */
OPAL_DECLSPEC opal_list_t *
opal_common_verbs_find_ports(const char *if_include,
                             const char *if_exclude,
                             int flags,
                             int verbose_stream);

/*
 * Trivial function to compute the bandwidth on an ibv_port.
 *
 * Will return OPAL_ERR_NOT_FOUND if it can't figure out the bandwidth
 * (and the bandwidth parameter value will be undefined).  Otherwise,
 * will return OPAL_SUCCESS and set bandwidth to an appropriate value.
 */
OPAL_DECLSPEC int
opal_common_verbs_port_bw(struct ibv_port_attr *port_attr,
                          uint32_t *bandwidth);

/*
 * Trivial function to switch on the verbs MTU enum and return a
 * numeric value.
 */
OPAL_DECLSPEC int
opal_common_verbs_mtu(struct ibv_port_attr *port_attr);

/*
 * Find the max_inline_data value for a given device
 */
OPAL_DECLSPEC int
opal_common_verbs_find_max_inline(struct ibv_device *device,
                                  struct ibv_context *context,
                                  struct ibv_pd *pd,
                                  uint32_t *max_inline_arg);

/*
 * Test a device to see if it can handle a specific QP type (RC and/or
 * UD).  Will return the logical AND if multiple types are specified
 * (e.g., if (RC|UD) are in flags, then will return OPAL_SUCCESS only
 * if *both* types can be created on the device).
 *
 * Flags can be the logical OR of OPAL_COMMON_VERBS_FLAGS_RC and/or
 * OPAL_COMMON_VERBS_FLAGS_UD.  All other values are ignored.
 */
OPAL_DECLSPEC int opal_common_verbs_qp_test(struct ibv_context *device_context,
                                            int flags);
/*
 * ibv_fork_init testing - if fork support is requested then ibv_fork_init
 * should be called right at the beginning of the verbs initialization flow, before ibv_create_* call.
 *
 * Known limitations:
 * If ibv_fork_init is called after ibv_create_* functions - it will have no effect.
 * OMPI initializes verbs many times during initialization in the following verbs components:
 *      oob/ud, btl/openib, mtl/mxm, pml/yalla, oshmem/ikrit, ompi/mca/coll/{fca,hcoll}
 *
 * So, ibv_fork_init should be called once, in the beginning of the init flow of every verb component
 * to proper request fork support.
 *
 */
int opal_common_verbs_fork_test(void);

END_C_DECLS

#endif

