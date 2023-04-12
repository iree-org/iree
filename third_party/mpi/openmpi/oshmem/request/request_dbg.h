/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef OSHMEM_REQUEST_DBG_H
#define OSHMEM_REQUEST_DBG_H

/*
 * This file contains definitions used by both OSHMEM and debugger plugins.
 * For more information on why we do this see the Notice to developers
 * comment at the top of the ompi_msgq_dll.c file.
 */

/**
 * Enum inidicating the type of the request
 */
typedef enum {
    OSHMEM_REQUEST_SPML, /**< MPI point-to-point request */
    OSHMEM_REQUEST_IO, /**< MPI-2 IO request */
    OSHMEM_REQUEST_GEN, /**< MPI-2 generalized request */
    OSHMEM_REQUEST_WIN,      /**< MPI-2 one-sided request */
    OSHMEM_REQUEST_COLL,     /**< MPI-3 non-blocking collectives request */
    OSHMEM_REQUEST_NULL, /**< NULL request */
    OSHMEM_REQUEST_NOOP, /**< A request that does nothing (e.g., to PROC_NULL) */
    OSHMEM_REQUEST_MAX /**< Maximum request type */
} oshmem_request_type_t;

/**
 * Enum indicating the state of the request
 */
typedef enum {
    /** Indicates that the request should not be progressed */
    OSHMEM_REQUEST_INVALID,
    /** A defined, but inactive request (i.e., it's valid, but should
     not be progressed) */
    OSHMEM_REQUEST_INACTIVE,
    /** A valid and progressing request */
    OSHMEM_REQUEST_ACTIVE,
    /** The request has been cancelled */
    OSHMEM_REQUEST_CANCELLED /* TODO: Not required */
} oshmem_request_state_t;

#endif
