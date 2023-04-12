/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2010-2013 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2018      Amazon.com, Inc. or its affiliates.  All Rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OPAL_MCA_IF_IF_H
#define OPAL_MCA_IF_IF_H

#include "opal_config.h"

#include <string.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <errno.h>
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_SYS_SOCKIO_H
#include <sys/sockio.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
#include <sys/ioctl.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NET_IF_H
#include <net/if.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_IFADDRS_H
#include <ifaddrs.h>
#endif

#include "opal/util/if.h"
#include "opal/mca/mca.h"
#include "opal/mca/base/base.h"

BEGIN_C_DECLS

/*
 * Define INADDR_NONE if we don't have it.  Solaris is the only system
 * where I have found that it does not exist, and the man page for
 * inet_addr() says that it returns -1 upon failure.  On Linux and
 * other systems with INADDR_NONE, it's just a #define to -1 anyway.
 * So just #define it to -1 here if it doesn't already exist.
 */

#if !defined(INADDR_NONE)
#define INADDR_NONE -1
#endif

#define DEFAULT_NUMBER_INTERFACES 10
#define MAX_IFCONF_SIZE 10 * 1024 * 1024


typedef struct opal_if_t {
    opal_list_item_t     super;
    char                if_name[IF_NAMESIZE];
    int                 if_index;
    uint16_t            if_kernel_index;
    uint16_t            af_family;
    int                 if_flags;
    int                 if_speed;
    struct sockaddr_storage  if_addr;
    uint32_t            if_mask;
    uint32_t            if_bandwidth;
    uint8_t             if_mac[6];
    int                 ifmtu; /* Can't use if_mtu because of a
                                  #define collision on some BSDs */
} opal_if_t;
OPAL_DECLSPEC OBJ_CLASS_DECLARATION(opal_if_t);


/* "global" list of available interfaces */
OPAL_DECLSPEC extern opal_list_t opal_if_list;

/* global flags */
OPAL_DECLSPEC extern bool opal_if_do_not_resolve;
OPAL_DECLSPEC extern bool opal_if_retain_loopback;

/**
 * Structure for if components.
 */
struct opal_if_base_component_2_0_0_t {
    /** MCA base component */
    mca_base_component_t component;
    /** MCA base data */
    mca_base_component_data_t component_data;
};
/**
 * Convenience typedef
 */
typedef struct opal_if_base_component_2_0_0_t opal_if_base_component_t;

/*
 * Macro for use in components that are of type if
 */
#define OPAL_IF_BASE_VERSION_2_0_0 \
    OPAL_MCA_BASE_VERSION_2_1_0("if", 2, 0, 0)

END_C_DECLS

#endif /* OPAL_MCA_IF_IF_H */
