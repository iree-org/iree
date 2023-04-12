/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef OSHMEM_OP_H
#define OSHMEM_OP_H

#include "oshmem_config.h"
#include "oshmem/types.h"
#include "oshmem/constants.h"

#include "oshmem/mca/scoll/scoll.h"

#include "opal/class/opal_list.h"
#include "opal/dss/dss_types.h"

#include "orte/types.h"

BEGIN_C_DECLS

/* ******************************************************************** */

/**
 * Corresponding to the types that we can reduce over.
 */
enum {
    OSHMEM_OP_TYPE_SHORT,       /** C integer: short */
    OSHMEM_OP_TYPE_INT,         /** C integer: int */
    OSHMEM_OP_TYPE_LONG,        /** C integer: long */
    OSHMEM_OP_TYPE_LLONG,       /** C integer: long long */
    OSHMEM_OP_TYPE_INT16_T,     /** C integer: int16_t */
    OSHMEM_OP_TYPE_INT32_T,     /** C integer: int32_t */
    OSHMEM_OP_TYPE_INT64_T,     /** C integer: int64_t */

    OSHMEM_OP_TYPE_FLOAT,       /** Floating point: float */
    OSHMEM_OP_TYPE_DOUBLE,      /** Floating point: double */
    OSHMEM_OP_TYPE_LDOUBLE,     /** Floating point: long double */

    OSHMEM_OP_TYPE_FCOMPLEX,    /** Complex: float */
    OSHMEM_OP_TYPE_DCOMPLEX,    /** Complex: double */

    OSHMEM_OP_TYPE_FINT2,       /** Fortran integer: int2 */
    OSHMEM_OP_TYPE_FINT4,       /** Fortran integer: int4 */
    OSHMEM_OP_TYPE_FINT8,       /** Fortran integer: int8 */
    OSHMEM_OP_TYPE_FREAL4,      /** Fortran integer: real4 */
    OSHMEM_OP_TYPE_FREAL8,      /** Fortran integer: real8 */
    OSHMEM_OP_TYPE_FREAL16,     /** Fortran integer: real16 */

    /** Maximum type */
    OSHMEM_OP_TYPE_NUMBER
};

/**
 * Supported reduce operations.
 */
enum {
    OSHMEM_OP_AND,      /** AND */
    OSHMEM_OP_OR,       /** OR */
    OSHMEM_OP_XOR,      /** XOR */
    OSHMEM_OP_MAX,      /** MAX */
    OSHMEM_OP_MIN,      /** MIN */
    OSHMEM_OP_SUM,      /** SUM */
    OSHMEM_OP_PROD,     /** PROD */

    /** Maximum operation */
    OSHMEM_OP_NUMBER
};

typedef void (oshmem_op_c_handler_fn_t)(void *, void *, int);

/**
 * Back-end type of OSHMEM reduction operations
 */
struct oshmem_op_t {
    opal_object_t               base;
    int                         id;             /**< index in global array */
    int                         op;             /**< operation type */
    int                         dt;             /**< datatype */
    size_t                      dt_size;        /**< datatype size */
    union {
        /** C handler function pointer */
        oshmem_op_c_handler_fn_t *c_fn;
    } o_func;
};
typedef struct oshmem_op_t oshmem_op_t;
OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(oshmem_op_t);

/* Bitwise AND */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_short;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_fint2;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_fint4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_fint8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_int16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_and_int64;

/* Bitwise OR */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_short;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_fint2;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_fint4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_fint8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_int16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_or_int64;

/* Bitwise XOR */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_short;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_fint2;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_fint4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_fint8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_int16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_xor_int64;

/* MAX */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_short;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_float;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_double;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_longdouble;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_fint2;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_fint4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_fint8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_freal4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_freal8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_freal16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_int16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_max_int64;

/* MIN */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_short;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_float;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_double;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_longdouble;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_fint2;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_fint4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_fint8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_freal4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_freal8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_freal16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_int16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_min_int64;

/* SUM */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_short;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_float;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_double;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_longdouble;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_complexf;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_complexd;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_fint2;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_fint4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_fint8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_freal4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_freal8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_freal16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_int16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_sum_int64;

/* PROD */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_short;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_float;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_double;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_longdouble;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_complexf;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_complexd;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_fint2;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_fint4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_fint8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_freal4;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_freal8;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_freal16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_int16;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_prod_int64;

/* SWAP */
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_swap_int;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_swap_long;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_swap_longlong;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_swap_int32;
OSHMEM_DECLSPEC extern oshmem_op_t* oshmem_op_swap_int64;

/**
 * Initialize the op interface.
 *
 * @returns OSHMEM_SUCCESS Upon success
 * @returns OSHMEM_ERROR Otherwise
 *
 * Invoked from oshmem_shmem_init(); sets up the op interface, creates
 * the predefined operations.
 */
int oshmem_op_init(void);

/**
 * Finalize the op interface.
 *
 * @returns OSHMEM_SUCCESS Always
 *
 * Invokes from oshmem_shmem_finalize(); tears down the op interface.
 */
int oshmem_op_finalize(void);

END_C_DECLS

#endif /* OSHMEM_OP_H */
