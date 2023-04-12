/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OMPI_F77_BINDINGS_H
#define OMPI_F77_BINDINGS_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/request/grequest.h"

#if OMPI_FORTRAN_CAPS
#define OMPI_GENERATE_F77_BINDINGS(upper_case, \
                                  lower_case, \
                                  single_underscore, \
                                  double_underscore, \
                                  wrapper_function, \
                                  signature, \
                                  params) \
            void upper_case signature { wrapper_function params; }
#elif OMPI_FORTRAN_PLAIN
#define OMPI_GENERATE_F77_BINDINGS(upper_case, \
                                  lower_case, \
                                  single_underscore, \
                                  double_underscore, \
                                  wrapper_function, \
                                  signature, \
                                  params) \
            void lower_case signature { wrapper_function params; }
#elif OMPI_FORTRAN_DOUBLE_UNDERSCORE
#define OMPI_GENERATE_F77_BINDINGS(upper_case, \
                                  lower_case, \
                                  single_underscore, \
                                  double_underscore, \
                                  wrapper_function, \
                                  signature, \
                                  params) \
            void double_underscore signature { wrapper_function params; }
#elif OMPI_FORTRAN_SINGLE_UNDERSCORE
#define OMPI_GENERATE_F77_BINDINGS(upper_case, \
                                  lower_case, \
                                  single_underscore, \
                                  double_underscore, \
                                  wrapper_function, \
                                  signature, \
                                  params) \
            void single_underscore signature { wrapper_function params; }
#else
#error Unrecognized Fortran name mangling scheme
#endif
/*
 * We maintain 2 separate sets of defines and prototypes. This ensures
 * that we can build MPI_* bindings or PMPI_* bindings as needed. The
 * top level always builds MPI_* bindings and bottom level will always
 * build PMPI_* bindings.  This means that top-level includes
 * "ompi/mpi/fortran/mpif-h/" .h files and lower-level includes
 * "ompi/mpi/fortran/mpif-h/profile" .h files.
 *
 * Both prototypes for all MPI / PMPI functions is moved into
 * prototypes_mpi.h.
 */

#include "ompi/mpi/fortran/mpif-h/prototypes_mpi.h"

#include "ompi/mpi/fortran/base/fint_2_int.h"

#endif /* OMPI_F77_BINDINGS_H */
