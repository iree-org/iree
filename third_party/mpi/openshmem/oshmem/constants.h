/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_CONSTANTS_H
#define OSHMEM_CONSTANTS_H

#include "orte/constants.h"
#include "oshmem/include/shmem.h"


#define OSHMEM_ERR_BASE   ORTE_ERR_MAX

/* error codes */
enum {
    /* Error codes inherited from ORTE/OPAL.  Still enum values so
       that we might get nice debugger help */
    OSHMEM_SUCCESS                  = ORTE_SUCCESS,

    OSHMEM_ERROR                    = ORTE_ERROR,
    OSHMEM_ERR_OUT_OF_RESOURCE      = ORTE_ERR_OUT_OF_RESOURCE,
    OSHMEM_ERR_TEMP_OUT_OF_RESOURCE = ORTE_ERR_TEMP_OUT_OF_RESOURCE,
    OSHMEM_ERR_RESOURCE_BUSY        = ORTE_ERR_RESOURCE_BUSY,
    OSHMEM_ERR_BAD_PARAM            = ORTE_ERR_BAD_PARAM,
    OSHMEM_ERR_FATAL                = ORTE_ERR_FATAL,
    OSHMEM_ERR_NOT_IMPLEMENTED      = ORTE_ERR_NOT_IMPLEMENTED,
    OSHMEM_ERR_NOT_SUPPORTED        = ORTE_ERR_NOT_SUPPORTED,
    OSHMEM_ERR_INTERUPTED           = ORTE_ERR_INTERUPTED,
    OSHMEM_ERR_WOULD_BLOCK          = ORTE_ERR_WOULD_BLOCK,
    OSHMEM_ERR_IN_ERRNO             = ORTE_ERR_IN_ERRNO,
    OSHMEM_ERR_UNREACH              = ORTE_ERR_UNREACH,
    OSHMEM_ERR_NOT_FOUND            = ORTE_ERR_NOT_FOUND,
    OSHMEM_EXISTS                   = ORTE_EXISTS, /* indicates that the specified object already exists */
    OSHMEM_ERR_TIMEOUT              = ORTE_ERR_TIMEOUT,
    OSHMEM_ERR_NOT_AVAILABLE        = ORTE_ERR_NOT_AVAILABLE,
    OSHMEM_ERR_PERM                 = ORTE_ERR_PERM,
    OSHMEM_ERR_VALUE_OUT_OF_BOUNDS  = ORTE_ERR_VALUE_OUT_OF_BOUNDS,
    OSHMEM_ERR_FILE_READ_FAILURE    = ORTE_ERR_FILE_READ_FAILURE,
    OSHMEM_ERR_FILE_WRITE_FAILURE   = ORTE_ERR_FILE_WRITE_FAILURE,
    OSHMEM_ERR_FILE_OPEN_FAILURE    = ORTE_ERR_FILE_OPEN_FAILURE,

    OSHMEM_ERR_RECV_LESS_THAN_POSTED      = ORTE_ERR_RECV_LESS_THAN_POSTED,
    OSHMEM_ERR_RECV_MORE_THAN_POSTED      = ORTE_ERR_RECV_MORE_THAN_POSTED,
    OSHMEM_ERR_NO_MATCH_YET               = ORTE_ERR_NO_MATCH_YET,
    OSHMEM_ERR_BUFFER                     = ORTE_ERR_BUFFER,
    OSHMEM_ERR_REQUEST                    = ORTE_ERR_REQUEST,
    OSHMEM_ERR_NO_CONNECTION_ALLOWED      = ORTE_ERR_NO_CONNECTION_ALLOWED,
    OSHMEM_ERR_CONNECTION_REFUSED         = ORTE_ERR_CONNECTION_REFUSED   ,
    OSHMEM_ERR_CONNECTION_FAILED          = ORTE_ERR_CONNECTION_FAILED,
    OSHMEM_PACK_MISMATCH                  = ORTE_ERR_PACK_MISMATCH,
    OSHMEM_ERR_PACK_FAILURE               = ORTE_ERR_PACK_FAILURE,
    OSHMEM_ERR_UNPACK_FAILURE             = ORTE_ERR_UNPACK_FAILURE,
    OSHMEM_ERR_COMM_FAILURE               = ORTE_ERR_COMM_FAILURE,
    OSHMEM_UNPACK_INADEQUATE_SPACE        = ORTE_ERR_UNPACK_INADEQUATE_SPACE,
    OSHMEM_UNPACK_READ_PAST_END_OF_BUFFER = ORTE_ERR_UNPACK_READ_PAST_END_OF_BUFFER,
    OSHMEM_ERR_TYPE_MISMATCH              = ORTE_ERR_TYPE_MISMATCH,
    OSHMEM_ERR_COMPARE_FAILURE            = ORTE_ERR_COMPARE_FAILURE,
    OSHMEM_ERR_COPY_FAILURE               = ORTE_ERR_COPY_FAILURE,
    OSHMEM_ERR_UNKNOWN_DATA_TYPE          = ORTE_ERR_UNKNOWN_DATA_TYPE,
    OSHMEM_ERR_DATA_TYPE_REDEF            = ORTE_ERR_DATA_TYPE_REDEF,
    OSHMEM_ERR_DATA_OVERWRITE_ATTEMPT     = ORTE_ERR_DATA_OVERWRITE_ATTEMPT
};


/* C datatypes */
/*
 * SHMEM_Init_thread constants
 * Do not change the order of these without also modifying mpif.h.in.
 */
enum {
  SHMEM_NULL	= 0,
  SHMEM_CHAR,
  SHMEM_UCHAR,
  SHMEM_SHORT,
  SHMEM_USHORT,
  SHMEM_INT,
  SHMEM_UINT,
  SHMEM_LONG,
  SHMEM_ULONG,
  SHMEM_LLONG,
  SHMEM_INT32_T,
  SHMEM_INT64_T,
  SHMEM_ULLONG,
  SHMEM_FLOAT,
  SHMEM_DOUBLE,
  SHMEM_LDOUBLE,

  SHMEM_FINT,
  SHMEM_FINT4,
  SHMEM_FINT8
};


/*
 * Miscellaneous constants
 */
#define SHMEM_ANY_SOURCE         -1                      /* match any source rank */
#define SHMEM_PROC_NULL          -2                      /* rank of null process */
#define SHMEM_UNDEFINED          -32766                  /* undefined stuff */


#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P) ((void)P)
#endif

#define OSHMEM_PREDEFINED_GLOBAL(type, global) ((type) ((void *) &(global)))

#if OPAL_WANT_MEMCHECKER
#define MEMCHECKER(x) do {       \
                x;                     \
       } while(0)
#else
#define MEMCHECKER(x)
#endif /* OPAL_WANT_MEMCHECKER */

#endif /* OSHMEM_CONSTANTS_H */

