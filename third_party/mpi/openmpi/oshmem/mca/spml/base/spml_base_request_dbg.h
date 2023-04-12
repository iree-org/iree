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
#ifndef MCA_SPML_BASE_REQUEST_DBG_H
#define MCA_SPML_BASE_REQUEST_DBG_H

/**
 * Type of request.
 */
typedef enum {
    MCA_SPML_REQUEST_NULL,
    MCA_SPML_REQUEST_PUT, /* Added */
    MCA_SPML_REQUEST_GET, /* Added */
    MCA_SPML_REQUEST_ATOMIC_CAS, /* Added */
    MCA_SPML_REQUEST_ATOMIC_FAAD /* Added */
} mca_spml_base_request_type_t;

#endif /* MCA_SPML_BASE_REQUEST_DBG_H */
