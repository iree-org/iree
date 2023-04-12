/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2013 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef SHMEM_FORTRAN_BINDINGS_H
#define SHMEM_FORTRAN_BINDINGS_H

#include "prototypes_shmem.h"
#include "ompi/mpi/fortran/base/fint_2_int.h"

#define SHMEM_GENERATE_FORTRAN_BINDINGS_FUNCTION(ret, \
        upper_case, \
        single_underscore, \
        double_underscore, \
        wrapper_function, \
        signature, \
        params) \
ret upper_case signature {return wrapper_function params; } \
ret single_underscore signature {return wrapper_function params; } \
ret double_underscore signature {return wrapper_function params; }

#define SHMEM_GENERATE_FORTRAN_BINDINGS_SUB(ret, \
        upper_case, \
        single_underscore, \
        double_underscore, \
        wrapper_function, \
        signature, \
        params) \
ret upper_case signature {wrapper_function params; } \
ret single_underscore signature {wrapper_function params; } \
ret double_underscore signature {wrapper_function params; }

#define FPTR_2_INT(ptr, size) ((size) == sizeof(uint64_t) ? *(uint64_t*)(ptr) : *(uint32_t*)(ptr))

#endif /*SHMEM_FORTRAN_BINDINGS_H*/
