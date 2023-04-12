/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2015 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2011-2013 Inria.  All rights reserved.
 * Copyright (c) 2011-2013 Universite Bordeaux 1
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2016-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * This file prototypes all MPI fortran functions in all four fortran
 * symbol conventions as well as all the internal real OMPI wrapper
 * functions (different from any of the four fortran symbol
 * conventions for clarity, at the cost of more typing for me...).
 * This file is included in the top-level build ONLY. The prototyping
 * is done ONLY for MPI_* bindings
 *
 * Zeroth, the OMPI wrapper functions, with a ompi_ prefix and _f
 * suffix.
 *
 * This is needed ONLY if the lower-level prototypes_pmpi.h has not
 * already been included.
 *
 * Note about function pointers: all function pointers are prototyped
 * here as (void*) rather than including the .h file that defines the
 * proper type (e.g., "op/op.h" defines ompi_op_fortran_handler_fn_t,
 * which is the function pointer type for fortran op callback
 * functions).  This is because there is no type checking coming in
 * from fortran, so why bother?  Also, including "op/op.h" (and
 * friends) makes the all the f77 bindings files dependant on these
 * files -- any change to any one of them will cause the recompilation
 * of the entire set of f77 bindings (ugh!).
 */

#ifndef OMPI_F77_PROTOTYPES_MPI_H
#define OMPI_F77_PROTOTYPES_MPI_H

#include "ompi_config.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/attribute/attribute.h"
#include "ompi/op/op.h"
#include "ompi/request/grequest.h"
#include "ompi/mpi/fortran/base/datarep.h"

BEGIN_C_DECLS

/* These are the prototypes for the "real" back-end fortran functions. */
#define PN2(ret, mixed_name, lower_name, upper_name, args) \
    /* Prototype the actual OMPI function */               \
    OMPI_DECLSPEC ret o##lower_name##_f args;              \
    /* Prototype the 4 versions of the MPI mpif.h name */  \
    OMPI_DECLSPEC ret lower_name args;                     \
    OMPI_DECLSPEC ret lower_name##_ args;                  \
    OMPI_DECLSPEC ret lower_name##__ args;                 \
    OMPI_DECLSPEC ret upper_name args;                     \
    /* Prototype the use mpi/use mpi_f08 names  */         \
    OMPI_DECLSPEC ret mixed_name##_f08 args;               \
    OMPI_DECLSPEC ret mixed_name##_f args;                 \
    /* Prototype the actual POMPI function */              \
    OMPI_DECLSPEC ret po##lower_name##_f args;             \
    /* Prototype the 4 versions of the PMPI mpif.h name */ \
    OMPI_DECLSPEC ret p##lower_name args;                  \
    OMPI_DECLSPEC ret p##lower_name##_ args;               \
    OMPI_DECLSPEC ret p##lower_name##__ args;              \
    OMPI_DECLSPEC ret P##upper_name args;                  \
    /* Prototype the use mpi/use mpi_f08 PMPI names  */    \
    OMPI_DECLSPEC ret P##mixed_name##_f08 args;            \
    OMPI_DECLSPEC ret P##mixed_name##_f args

PN2(void, MPI_Abort, mpi_abort, MPI_ABORT, (MPI_Fint *comm, MPI_Fint *errorcode, MPI_Fint *ierr));
PN2(void, MPI_Accumulate, mpi_accumulate, MPI_ACCUMULATE, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Add_error_class, mpi_add_error_class, MPI_ADD_ERROR_CLASS, (MPI_Fint *errorclass, MPI_Fint *ierr));
PN2(void, MPI_Add_error_code, mpi_add_error_code, MPI_ADD_ERROR_CODE, (MPI_Fint *errorclass, MPI_Fint *errorcode, MPI_Fint *ierr));
PN2(void, MPI_Add_error_string, mpi_add_error_string, MPI_ADD_ERROR_STRING, (MPI_Fint *errorcode, char *string, MPI_Fint *ierr, int l));
PN2(void, MPI_Address, mpi_address, MPI_ADDRESS, (char *location, MPI_Fint *address, MPI_Fint *ierr));
PN2(MPI_Aint, MPI_Aint_add, mpi_aint_add, MPI_AINT_ADD, (MPI_Aint *base, MPI_Aint *diff));
PN2(MPI_Aint, MPI_Aint_diff, mpi_aint_diff, MPI_AINT_DIFF, (MPI_Aint *addr1, MPI_Aint *addr2));
PN2(void, MPI_Allgather, mpi_allgather, MPI_ALLGATHER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Allgatherv, mpi_allgatherv, MPI_ALLGATHERV, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Alloc_mem, mpi_alloc_mem, MPI_ALLOC_MEM, (MPI_Aint *size, MPI_Fint *info, char *baseptr, MPI_Fint *ierr));
/* Extra Alloc_mem prototype for the _cptr variant added in MPI-3.0 errata */
PN2(void, MPI_Alloc_mem_cptr, mpi_alloc_mem_cptr, MPI_ALLOC_MEM_CPTR, (MPI_Aint *size, MPI_Fint *info, char *baseptr, MPI_Fint *ierr));
PN2(void, MPI_Allreduce, mpi_allreduce, MPI_ALLREDUCE, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Alltoall, mpi_alltoall, MPI_ALLTOALL, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Alltoallv, mpi_alltoallv, MPI_ALLTOALLV, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Alltoallw, mpi_alltoallw, MPI_ALLTOALLW, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Attr_delete, mpi_attr_delete, MPI_ATTR_DELETE, (MPI_Fint *comm, MPI_Fint *keyval, MPI_Fint *ierr));
PN2(void, MPI_Attr_get, mpi_attr_get, MPI_ATTR_GET, (MPI_Fint *comm, MPI_Fint *keyval, MPI_Fint *attribute_val, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Attr_put, mpi_attr_put, MPI_ATTR_PUT, (MPI_Fint *comm, MPI_Fint *keyval, MPI_Fint *attribute_val, MPI_Fint *ierr));
PN2(void, MPI_Barrier, mpi_barrier, MPI_BARRIER, (MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Bcast, mpi_bcast, MPI_BCAST, (char *buffer, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Bsend, mpi_bsend, MPI_BSEND, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Bsend_init, mpi_bsend_init, MPI_BSEND_INIT, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Buffer_attach, mpi_buffer_attach, MPI_BUFFER_ATTACH, (char *buffer, MPI_Fint *size, MPI_Fint *ierr));
PN2(void, MPI_Buffer_detach, mpi_buffer_detach, MPI_BUFFER_DETACH, (char *buffer, MPI_Fint *size, MPI_Fint *ierr));
PN2(void, MPI_Cancel, mpi_cancel, MPI_CANCEL, (MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Cart_coords, mpi_cart_coords, MPI_CART_COORDS, (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *maxdims, MPI_Fint *coords, MPI_Fint *ierr));
PN2(void, MPI_Cart_create, mpi_cart_create, MPI_CART_CREATE, (MPI_Fint *old_comm, MPI_Fint *ndims, MPI_Fint *dims, ompi_fortran_logical_t *periods, ompi_fortran_logical_t *reorder, MPI_Fint *comm_cart, MPI_Fint *ierr));
PN2(void, MPI_Cart_get, mpi_cart_get, MPI_CART_GET, (MPI_Fint *comm, MPI_Fint *maxdims, MPI_Fint *dims, ompi_fortran_logical_t *periods, MPI_Fint *coords, MPI_Fint *ierr));
PN2(void, MPI_Cart_map, mpi_cart_map, MPI_CART_MAP, (MPI_Fint *comm, MPI_Fint *ndims, MPI_Fint *dims, ompi_fortran_logical_t *periods, MPI_Fint *newrank, MPI_Fint *ierr));
PN2(void, MPI_Cart_rank, mpi_cart_rank, MPI_CART_RANK, (MPI_Fint *comm, MPI_Fint *coords, MPI_Fint *rank, MPI_Fint *ierr));
PN2(void, MPI_Cart_shift, mpi_cart_shift, MPI_CART_SHIFT, (MPI_Fint *comm, MPI_Fint *direction, MPI_Fint *disp, MPI_Fint *rank_source, MPI_Fint *rank_dest, MPI_Fint *ierr));
PN2(void, MPI_Cart_sub, mpi_cart_sub, MPI_CART_SUB, (MPI_Fint *comm, ompi_fortran_logical_t *remain_dims, MPI_Fint *new_comm, MPI_Fint *ierr));
PN2(void, MPI_Cartdim_get, mpi_cartdim_get, MPI_CARTDIM_GET, (MPI_Fint *comm, MPI_Fint *ndims, MPI_Fint *ierr));
PN2(void, MPI_Close_port, mpi_close_port, MPI_CLOSE_PORT, (char *port_name, MPI_Fint *ierr, int port_name_len));
PN2(void, MPI_Comm_accept, mpi_comm_accept, MPI_COMM_ACCEPT, (char *port_name, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *newcomm, MPI_Fint *ierr, int port_name_len));
PN2(void, MPI_Comm_call_errhandler, mpi_comm_call_errhandler, MPI_COMM_CALL_ERRHANDLER, (MPI_Fint *comm, MPI_Fint *errorcode, MPI_Fint *ierr));
PN2(void, MPI_Comm_compare, mpi_comm_compare, MPI_COMM_COMPARE, (MPI_Fint *comm1, MPI_Fint *comm2, MPI_Fint *result, MPI_Fint *ierr));
PN2(void, MPI_Comm_connect, mpi_comm_connect, MPI_COMM_CONNECT, (char *port_name, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *newcomm, MPI_Fint *ierr, int port_name_len));
PN2(void, MPI_Comm_create_errhandler, mpi_comm_create_errhandler, MPI_COMM_CREATE_ERRHANDLER, (ompi_errhandler_fortran_handler_fn_t* function, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Comm_create_keyval, mpi_comm_create_keyval, MPI_COMM_CREATE_KEYVAL, (ompi_aint_copy_attr_function* comm_copy_attr_fn, ompi_aint_delete_attr_function* comm_delete_attr_fn, MPI_Fint *comm_keyval, MPI_Aint *extra_state, MPI_Fint *ierr));
PN2(void, MPI_Comm_create, mpi_comm_create, MPI_COMM_CREATE, (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *newcomm, MPI_Fint *ierr));
PN2(void, MPI_Comm_create_group, mpi_comm_create_group, MPI_COMM_CREATE_GROUP, (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *tag, MPI_Fint *newcomm, MPI_Fint *ierr));
PN2(void, MPI_Comm_delete_attr, mpi_comm_delete_attr, MPI_COMM_DELETE_ATTR, (MPI_Fint *comm, MPI_Fint *comm_keyval, MPI_Fint *ierr));
PN2(void, MPI_Comm_disconnect, mpi_comm_disconnect, MPI_COMM_DISCONNECT, (MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Comm_dup, mpi_comm_dup, MPI_COMM_DUP, (MPI_Fint *comm, MPI_Fint *newcomm, MPI_Fint *ierr));
PN2(void, MPI_Comm_dup_with_info, mpi_comm_dup_with_info, MPI_COMM_DUP_WITH_INFO, (MPI_Fint *comm, MPI_Fint *info, MPI_Fint *newcomm, MPI_Fint *ierr));
PN2(void, MPI_Comm_idup, mpi_comm_idup, MPI_COMM_IDUP, (MPI_Fint *comm, MPI_Fint *newcomm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Comm_free_keyval, mpi_comm_free_keyval, MPI_COMM_FREE_KEYVAL, (MPI_Fint *comm_keyval, MPI_Fint *ierr));
PN2(void, MPI_Comm_free, mpi_comm_free, MPI_COMM_FREE, (MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Comm_get_attr, mpi_comm_get_attr, MPI_COMM_GET_ATTR, (MPI_Fint *comm, MPI_Fint *comm_keyval, MPI_Aint *attribute_val, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Comm_get_info, mpi_comm_get_info, MPI_COMM_GET_INFO, (MPI_Fint *comm, MPI_Fint *info_user, MPI_Fint *ierr));
PN2(void, MPI_Comm_get_errhandler, mpi_comm_get_errhandler, MPI_COMM_GET_ERRHANDLER, (MPI_Fint *comm, MPI_Fint *erhandler, MPI_Fint *ierr));
PN2(void, MPI_Comm_get_name, mpi_comm_get_name, MPI_COMM_GET_NAME, (MPI_Fint *comm, char *comm_name, MPI_Fint *resultlen, MPI_Fint *ierr, int name_len));
PN2(void, MPI_Comm_get_parent, mpi_comm_get_parent, MPI_COMM_GET_PARENT, (MPI_Fint *parent, MPI_Fint *ierr));
PN2(void, MPI_Comm_group, mpi_comm_group, MPI_COMM_GROUP, (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *ierr));
PN2(void, MPI_Comm_join, mpi_comm_join, MPI_COMM_JOIN, (MPI_Fint *fd, MPI_Fint *intercomm, MPI_Fint *ierr));
PN2(void, MPI_Comm_rank, mpi_comm_rank, MPI_COMM_RANK, (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *ierr));
PN2(void, MPI_Comm_remote_group, mpi_comm_remote_group, MPI_COMM_REMOTE_GROUP, (MPI_Fint *comm, MPI_Fint *group, MPI_Fint *ierr));
PN2(void, MPI_Comm_remote_size, mpi_comm_remote_size, MPI_COMM_REMOTE_SIZE, (MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierr));
PN2(void, MPI_Comm_set_attr, mpi_comm_set_attr, MPI_COMM_SET_ATTR, (MPI_Fint *comm, MPI_Fint *comm_keyval, MPI_Aint *attribute_val, MPI_Fint *ierr));
PN2(void, MPI_Comm_set_info, mpi_comm_set_info, MPI_COMM_SET_INFO, (MPI_Fint *comm, MPI_Fint *info, MPI_Fint *ierr));
PN2(void, MPI_Comm_set_errhandler, mpi_comm_set_errhandler, MPI_COMM_SET_ERRHANDLER, (MPI_Fint *comm, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Comm_set_name, mpi_comm_set_name, MPI_COMM_SET_NAME, (MPI_Fint *comm, char *comm_name, MPI_Fint *ierr, int name_len));
PN2(void, MPI_Comm_size, mpi_comm_size, MPI_COMM_SIZE, (MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierr));
PN2(void, MPI_Comm_spawn, mpi_comm_spawn, MPI_COMM_SPAWN, (char *command, char *argv, MPI_Fint *maxprocs, MPI_Fint *info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierr, int command_len, int argv_len));
PN2(void, MPI_Comm_spawn_multiple, mpi_comm_spawn_multiple, MPI_COMM_SPAWN_MULTIPLE, (MPI_Fint *count, char *array_of_commands, char *array_of_argv, MPI_Fint *array_of_maxprocs, MPI_Fint *array_of_info, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *intercomm, MPI_Fint *array_of_errcodes, MPI_Fint *ierr, int cmd_len, int argv_len));
PN2(void, MPI_Comm_split, mpi_comm_split, MPI_COMM_SPLIT, (MPI_Fint *comm, MPI_Fint *color, MPI_Fint *key, MPI_Fint *newcomm, MPI_Fint *ierr));
PN2(void, MPI_Comm_split_type, mpi_comm_split_type, MPI_COMM_SPLIT_TYPE, (MPI_Fint *comm, MPI_Fint *split_type, MPI_Fint *key, MPI_Fint *info, MPI_Fint *newcomm, MPI_Fint *ierr));
PN2(void, MPI_Comm_test_inter, mpi_comm_test_inter, MPI_COMM_TEST_INTER, (MPI_Fint *comm, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Compare_and_swap, mpi_compare_and_swap, MPI_COMPARE_AND_SWAP, (char *origin_addr, char *compare_addr, char *result_addr, MPI_Fint *datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Dims_create, mpi_dims_create, MPI_DIMS_CREATE, (MPI_Fint *nnodes, MPI_Fint *ndims, MPI_Fint *dims, MPI_Fint *ierr));
PN2(void, MPI_Dist_graph_create, mpi_dist_graph_create, MPI_DIST_GRAPH_CREATE, (MPI_Fint *comm_old, MPI_Fint *n, MPI_Fint *sources, MPI_Fint *degrees, MPI_Fint *destinations, MPI_Fint *weights, MPI_Fint *info,  ompi_fortran_logical_t *reorder, MPI_Fint *comm_graph,  MPI_Fint *ierr));
PN2(void, MPI_Dist_graph_create_adjacent, mpi_dist_graph_create_adjacent, MPI_DIST_GRAPH_CREATE_ADJACENT, (MPI_Fint *comm_old, MPI_Fint *indegree,  MPI_Fint *sources, MPI_Fint *sourceweights, MPI_Fint *outdegree,  MPI_Fint *destinations, MPI_Fint *destweights, MPI_Fint *info, ompi_fortran_logical_t *reorder, MPI_Fint *comm_graph, MPI_Fint *ierr));
PN2(void, MPI_Dist_graph_neighbors, mpi_dist_graph_neighbors, MPI_DIST_GRAPH_NEIGHBORS, (MPI_Fint* comm, MPI_Fint* maxindegree, MPI_Fint* sources, MPI_Fint* sourceweights, MPI_Fint* maxoutdegree, MPI_Fint* destinations, MPI_Fint* destweights, MPI_Fint *ierr));
PN2(void, MPI_Dist_graph_neighbors_count, mpi_dist_graph_neighbors_count, MPI_DIST_GRAPH_NEIGHBORS_COUNT, (MPI_Fint *comm, MPI_Fint *inneighbors, MPI_Fint *outneighbors, ompi_fortran_logical_t *weighted, MPI_Fint *ierr));
PN2(void, MPI_Errhandler_create, mpi_errhandler_create, MPI_ERRHANDLER_CREATE, (ompi_errhandler_fortran_handler_fn_t* function, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Errhandler_free, mpi_errhandler_free, MPI_ERRHANDLER_FREE, (MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Errhandler_get, mpi_errhandler_get, MPI_ERRHANDLER_GET, (MPI_Fint *comm, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Errhandler_set, mpi_errhandler_set, MPI_ERRHANDLER_SET, (MPI_Fint *comm, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Error_class, mpi_error_class, MPI_ERROR_CLASS, (MPI_Fint *errorcode, MPI_Fint *errorclass, MPI_Fint *ierr));
PN2(void, MPI_Error_string, mpi_error_string, MPI_ERROR_STRING, (MPI_Fint *errorcode, char *string, MPI_Fint *resultlen, MPI_Fint *ierr, int string_len));
PN2(void, MPI_Exscan, mpi_exscan, MPI_EXSCAN, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_F_sync_reg, mpi_f_sync_reg, MPI_F_SYNC_REG, (char *buf));
PN2(void, MPI_Fetch_and_op, mpi_fetch_and_op, MPI_FETCH_AND_OP, (char *origin_addr, char *result_addr, MPI_Fint *datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *op, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_File_call_errhandler, mpi_file_call_errhandler, MPI_FILE_CALL_ERRHANDLER, (MPI_Fint *fh, MPI_Fint *errorcode, MPI_Fint *ierr));
PN2(void, MPI_File_create_errhandler, mpi_file_create_errhandler, MPI_FILE_CREATE_ERRHANDLER, (ompi_errhandler_fortran_handler_fn_t* function, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_File_set_errhandler, mpi_file_set_errhandler, MPI_FILE_SET_ERRHANDLER, (MPI_Fint *file, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_File_get_errhandler, mpi_file_get_errhandler, MPI_FILE_GET_ERRHANDLER, (MPI_Fint *file, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_File_open, mpi_file_open, MPI_FILE_OPEN, (MPI_Fint *comm, char *filename, MPI_Fint *amode, MPI_Fint *info, MPI_Fint *fh, MPI_Fint *ierr, int name_len));
PN2(void, MPI_File_close, mpi_file_close, MPI_FILE_CLOSE, (MPI_Fint *fh, MPI_Fint *ierr));
PN2(void, MPI_File_delete, mpi_file_delete, MPI_FILE_DELETE, (char *filename, MPI_Fint *info, MPI_Fint *ierr, int filename_len));
PN2(void, MPI_File_set_size, mpi_file_set_size, MPI_FILE_SET_SIZE, (MPI_Fint *fh, MPI_Offset *size, MPI_Fint *ierr));
PN2(void, MPI_File_preallocate, mpi_file_preallocate, MPI_FILE_PREALLOCATE, (MPI_Fint *fh, MPI_Offset *size, MPI_Fint *ierr));
PN2(void, MPI_File_get_size, mpi_file_get_size, MPI_FILE_GET_SIZE, (MPI_Fint *fh, MPI_Offset *size, MPI_Fint *ierr));
PN2(void, MPI_File_get_group, mpi_file_get_group, MPI_FILE_GET_GROUP, (MPI_Fint *fh, MPI_Fint *group, MPI_Fint *ierr));
PN2(void, MPI_File_get_amode, mpi_file_get_amode, MPI_FILE_GET_AMODE, (MPI_Fint *fh, MPI_Fint *amode, MPI_Fint *ierr));
PN2(void, MPI_File_set_info, mpi_file_set_info, MPI_FILE_SET_INFO, (MPI_Fint *fh, MPI_Fint *info, MPI_Fint *ierr));
PN2(void, MPI_File_get_info, mpi_file_get_info, MPI_FILE_GET_INFO, (MPI_Fint *fh, MPI_Fint *info_used, MPI_Fint *ierr));
PN2(void, MPI_File_set_view, mpi_file_set_view, MPI_FILE_SET_VIEW, (MPI_Fint *fh, MPI_Offset *disp, MPI_Fint *etype, MPI_Fint *filetype, char *datarep, MPI_Fint *info, MPI_Fint *ierr, int datarep_len));
PN2(void, MPI_File_get_view, mpi_file_get_view, MPI_FILE_GET_VIEW, (MPI_Fint *fh, MPI_Offset *disp, MPI_Fint *etype, MPI_Fint *filetype, char *datarep, MPI_Fint *ierr, int datarep_len));
PN2(void, MPI_File_read_at, mpi_file_read_at, MPI_FILE_READ_AT, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_read_at_all, mpi_file_read_at_all, MPI_FILE_READ_AT_ALL, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_at, mpi_file_write_at, MPI_FILE_WRITE_AT, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_at_all, mpi_file_write_at_all, MPI_FILE_WRITE_AT_ALL, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_iread_at, mpi_file_iread_at, MPI_FILE_IREAD_AT, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_iwrite_at, mpi_file_iwrite_at, MPI_FILE_IWRITE_AT, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_iread_at_all, mpi_file_iread_at_all, MPI_FILE_IREAD_AT_ALL, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_iwrite_at_all, mpi_file_iwrite_at_all, MPI_FILE_IWRITE_AT_ALL, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_read, mpi_file_read, MPI_FILE_READ, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_read_all, mpi_file_read_all, MPI_FILE_READ_ALL, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write, mpi_file_write, MPI_FILE_WRITE, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_all, mpi_file_write_all, MPI_FILE_WRITE_ALL, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_iread, mpi_file_iread, MPI_FILE_IREAD, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_iwrite, mpi_file_iwrite, MPI_FILE_IWRITE, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_iread_all, mpi_file_iread_all, MPI_FILE_IREAD_ALL, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_iwrite_all, mpi_file_iwrite_all, MPI_FILE_IWRITE_ALL, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_seek, mpi_file_seek, MPI_FILE_SEEK, (MPI_Fint *fh, MPI_Offset *offset, MPI_Fint *whence, MPI_Fint *ierr));
PN2(void, MPI_File_get_position, mpi_file_get_position, MPI_FILE_GET_POSITION, (MPI_Fint *fh, MPI_Offset *offset, MPI_Fint *ierr));
PN2(void, MPI_File_get_byte_offset, mpi_file_get_byte_offset, MPI_FILE_GET_BYTE_OFFSET, (MPI_Fint *fh, MPI_Offset *offset, MPI_Offset *disp, MPI_Fint *ierr));
PN2(void, MPI_File_read_shared, mpi_file_read_shared, MPI_FILE_READ_SHARED, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_shared, mpi_file_write_shared, MPI_FILE_WRITE_SHARED, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_iread_shared, mpi_file_iread_shared, MPI_FILE_IREAD_SHARED, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_iwrite_shared, mpi_file_iwrite_shared, MPI_FILE_IWRITE_SHARED, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_File_read_ordered, mpi_file_read_ordered, MPI_FILE_READ_ORDERED, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_ordered, mpi_file_write_ordered, MPI_FILE_WRITE_ORDERED, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_seek_shared, mpi_file_seek_shared, MPI_FILE_SEEK_SHARED, (MPI_Fint *fh, MPI_Offset *offset, MPI_Fint *whence, MPI_Fint *ierr));
PN2(void, MPI_File_get_position_shared, mpi_file_get_position_shared, MPI_FILE_GET_POSITION_SHARED, (MPI_Fint *fh, MPI_Offset *offset, MPI_Fint *ierr));
PN2(void, MPI_File_read_at_all_begin, mpi_file_read_at_all_begin, MPI_FILE_READ_AT_ALL_BEGIN, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *ierr));
PN2(void, MPI_File_read_at_all_end, mpi_file_read_at_all_end, MPI_FILE_READ_AT_ALL_END, (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_at_all_begin, mpi_file_write_at_all_begin, MPI_FILE_WRITE_AT_ALL_BEGIN, (MPI_Fint *fh, MPI_Offset *offset, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *ierr));
PN2(void, MPI_File_write_at_all_end, mpi_file_write_at_all_end, MPI_FILE_WRITE_AT_ALL_END, (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_read_all_begin, mpi_file_read_all_begin, MPI_FILE_READ_ALL_BEGIN, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *ierr));
PN2(void, MPI_File_read_all_end, mpi_file_read_all_end, MPI_FILE_READ_ALL_END, (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_all_begin, mpi_file_write_all_begin, MPI_FILE_WRITE_ALL_BEGIN, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *ierr));
PN2(void, MPI_File_write_all_end, mpi_file_write_all_end, MPI_FILE_WRITE_ALL_END, (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_read_ordered_begin, mpi_file_read_ordered_begin, MPI_FILE_READ_ORDERED_BEGIN, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *ierr));
PN2(void, MPI_File_read_ordered_end, mpi_file_read_ordered_end, MPI_FILE_READ_ORDERED_END, (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_write_ordered_begin, mpi_file_write_ordered_begin, MPI_FILE_WRITE_ORDERED_BEGIN, (MPI_Fint *fh, char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *ierr));
PN2(void, MPI_File_write_ordered_end, mpi_file_write_ordered_end, MPI_FILE_WRITE_ORDERED_END, (MPI_Fint *fh, char *buf, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_File_get_type_extent, mpi_file_get_type_extent, MPI_FILE_GET_TYPE_EXTENT, (MPI_Fint *fh, MPI_Fint *datatype, MPI_Aint *extent, MPI_Fint *ierr));
PN2(void, MPI_File_set_atomicity, mpi_file_set_atomicity, MPI_FILE_SET_ATOMICITY, (MPI_Fint *fh, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_File_get_atomicity, mpi_file_get_atomicity, MPI_FILE_GET_ATOMICITY, (MPI_Fint *fh, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_File_sync, mpi_file_sync, MPI_FILE_SYNC, (MPI_Fint *fh, MPI_Fint *ierr));
PN2(void, MPI_Finalize, mpi_finalize, MPI_FINALIZE, (MPI_Fint *ierr));
PN2(void, MPI_Finalized, mpi_finalized, MPI_FINALIZED, (ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Free_mem, mpi_free_mem, MPI_FREE_MEM, (char *base, MPI_Fint *ierr));
PN2(void, MPI_Gather, mpi_gather, MPI_GATHER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Gatherv, mpi_gatherv, MPI_GATHERV, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Get_accumulate, mpi_get_accumulate, MPI_GET_ACCUMULATE, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, char *result_addr, MPI_Fint *result_count, MPI_Fint *result_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Get_address, mpi_get_address, MPI_GET_ADDRESS, (char *location, MPI_Aint *address, MPI_Fint *ierr));
PN2(void, MPI_Get_count, mpi_get_count, MPI_GET_COUNT, (MPI_Fint *status, MPI_Fint *datatype, MPI_Fint *count, MPI_Fint *ierr));
PN2(void, MPI_Get_elements, mpi_get_elements, MPI_GET_ELEMENTS, (MPI_Fint *status, MPI_Fint *datatype, MPI_Fint *count, MPI_Fint *ierr));
PN2(void, MPI_Get_elements_x, mpi_get_elements_x, MPI_GET_ELEMENTS_X, (MPI_Fint *status, MPI_Fint *datatype, MPI_Count *count, MPI_Fint *ierr));
PN2(void, MPI_Get, mpi_get, MPI_GET, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Get_library_version, mpi_get_library_version, MPI_GET_LIBRARY_VERSION, (char *version, MPI_Fint *resultlen, MPI_Fint *ierr, MPI_Fint version_len));
PN2(void, MPI_Get_processor_name, mpi_get_processor_name, MPI_GET_PROCESSOR_NAME, (char *name, MPI_Fint *resultlen, MPI_Fint *ierr, int name_len));
PN2(void, MPI_Get_version, mpi_get_version, MPI_GET_VERSION, (MPI_Fint *version, MPI_Fint *subversion, MPI_Fint *ierr));
PN2(void, MPI_Graph_create, mpi_graph_create, MPI_GRAPH_CREATE, (MPI_Fint *comm_old, MPI_Fint *nnodes, MPI_Fint *index, MPI_Fint *edges, ompi_fortran_logical_t *reorder, MPI_Fint *comm_graph, MPI_Fint *ierr));
PN2(void, MPI_Graph_get, mpi_graph_get, MPI_GRAPH_GET, (MPI_Fint *comm, MPI_Fint *maxindex, MPI_Fint *maxedges, MPI_Fint *index, MPI_Fint *edges, MPI_Fint *ierr));
PN2(void, MPI_Graph_map, mpi_graph_map, MPI_GRAPH_MAP, (MPI_Fint *comm, MPI_Fint *nnodes, MPI_Fint *index, MPI_Fint *edges, MPI_Fint *newrank, MPI_Fint *ierr));
PN2(void, MPI_Graph_neighbors_count, mpi_graph_neighbors_count, MPI_GRAPH_NEIGHBORS_COUNT, (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *nneighbors, MPI_Fint *ierr));
PN2(void, MPI_Graph_neighbors, mpi_graph_neighbors, MPI_GRAPH_NEIGHBORS, (MPI_Fint *comm, MPI_Fint *rank, MPI_Fint *maxneighbors, MPI_Fint *neighbors, MPI_Fint *ierr));
PN2(void, MPI_Graphdims_get, mpi_graphdims_get, MPI_GRAPHDIMS_GET, (MPI_Fint *comm, MPI_Fint *nnodes, MPI_Fint *nedges, MPI_Fint *ierr));
PN2(void, MPI_Grequest_complete, mpi_grequest_complete, MPI_GREQUEST_COMPLETE, (MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Grequest_start, mpi_grequest_start, MPI_GREQUEST_START, (MPI_F_Grequest_query_function* query_fn, MPI_F_Grequest_free_function* free_fn, MPI_F_Grequest_cancel_function* cancel_fn, MPI_Aint *extra_state, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Group_compare, mpi_group_compare, MPI_GROUP_COMPARE, (MPI_Fint *group1, MPI_Fint *group2, MPI_Fint *result, MPI_Fint *ierr));
PN2(void, MPI_Group_difference, mpi_group_difference, MPI_GROUP_DIFFERENCE, (MPI_Fint *group1, MPI_Fint *group2, MPI_Fint *newgroup, MPI_Fint *ierr));
PN2(void, MPI_Group_excl, mpi_group_excl, MPI_GROUP_EXCL, (MPI_Fint *group, MPI_Fint *n, MPI_Fint *ranks, MPI_Fint *newgroup, MPI_Fint *ierr));
PN2(void, MPI_Group_free, mpi_group_free, MPI_GROUP_FREE, (MPI_Fint *group, MPI_Fint *ierr));
PN2(void, MPI_Group_incl, mpi_group_incl, MPI_GROUP_INCL, (MPI_Fint *group, MPI_Fint *n, MPI_Fint *ranks, MPI_Fint *newgroup, MPI_Fint *ierr));
PN2(void, MPI_Group_intersection, mpi_group_intersection, MPI_GROUP_INTERSECTION, (MPI_Fint *group1, MPI_Fint *group2, MPI_Fint *newgroup, MPI_Fint *ierr));
PN2(void, MPI_Group_range_excl, mpi_group_range_excl, MPI_GROUP_RANGE_EXCL, (MPI_Fint *group, MPI_Fint *n, MPI_Fint ranges[][3], MPI_Fint *newgroup, MPI_Fint *ierr));
PN2(void, MPI_Group_range_incl, mpi_group_range_incl, MPI_GROUP_RANGE_INCL, (MPI_Fint *group, MPI_Fint *n, MPI_Fint ranges[][3], MPI_Fint *newgroup, MPI_Fint *ierr));
PN2(void, MPI_Group_rank, mpi_group_rank, MPI_GROUP_RANK, (MPI_Fint *group, MPI_Fint *rank, MPI_Fint *ierr));
PN2(void, MPI_Group_size, mpi_group_size, MPI_GROUP_SIZE, (MPI_Fint *group, MPI_Fint *size, MPI_Fint *ierr));
PN2(void, MPI_Group_translate_ranks, mpi_group_translate_ranks, MPI_GROUP_TRANSLATE_RANKS, (MPI_Fint *group1, MPI_Fint *n, MPI_Fint *ranks1, MPI_Fint *group2, MPI_Fint *ranks2, MPI_Fint *ierr));
PN2(void, MPI_Group_union, mpi_group_union, MPI_GROUP_UNION, (MPI_Fint *group1, MPI_Fint *group2, MPI_Fint *newgroup, MPI_Fint *ierr));
PN2(void, MPI_Iallgather, mpi_iallgather, MPI_IALLGATHER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Iallgatherv, mpi_iallgatherv, MPI_IALLGATHERV, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Iallreduce, mpi_iallreduce, MPI_IALLREDUCE, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ialltoall, mpi_ialltoall, MPI_IALLTOALL, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ialltoallv, mpi_ialltoallv, MPI_IALLTOALLV, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ialltoallw, mpi_ialltoallw, MPI_IALLTOALLW, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ibarrier, mpi_ibarrier, MPI_IBARRIER, (MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ibcast, mpi_ibcast, MPI_IBCAST, (char *buffer, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ibsend, mpi_ibsend, MPI_IBSEND, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Iexscan, mpi_iexscan, MPI_IEXSCAN, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Igather, mpi_igather, MPI_IGATHER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Igatherv, mpi_igatherv, MPI_IGATHERV, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Improbe, mpi_improbe, MPI_IMPROBE, (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, ompi_fortran_logical_t *flag, MPI_Fint *message, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Imrecv,mpi_imrecv, MPI_IMRECV, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *message, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ineighbor_allgather, mpi_ineighbor_allgather, MPI_INEIGHBOR_ALLGATHER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ineighbor_allgatherv, mpi_ineighbor_allgatherv, MPI_INEIGHBOR_ALLGATHERV, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ineighbor_alltoall, mpi_ineighbor_alltoall, MPI_INEIGHBOR_ALLTOALL, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ineighbor_alltoallv, mpi_ineighbor_alltoallv, MPI_INEIGHBOR_ALLTOALLV, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ineighbor_alltoallw, mpi_ineighbor_alltoallw, MPI_INEIGHBOR_ALLTOALLW, (char *sendbuf, MPI_Fint *sendcounts, MPI_Aint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Aint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ireduce, mpi_ireduce, MPI_IREDUCE, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ireduce_scatter, mpi_ireduce_scatter, MPI_IREDUCE_SCATTER, (char *sendbuf, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ireduce_scatter_block, mpi_ireduce_scatter_block, MPI_IREDUCE_SCATTER_BLOCK, (char *sendbuf, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Iscan, mpi_iscan, MPI_ISCAN, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Iscatter, mpi_iscatter, MPI_ISCATTER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Iscatterv, mpi_iscatterv, MPI_ISCATTERV, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *displs, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Info_create, mpi_info_create, MPI_INFO_CREATE, (MPI_Fint *info, MPI_Fint *ierr));
PN2(void, MPI_Info_delete, mpi_info_delete, MPI_INFO_DELETE, (MPI_Fint *info, char *key, MPI_Fint *ierr, int key_len));
PN2(void, MPI_Info_dup, mpi_info_dup, MPI_INFO_DUP, (MPI_Fint *info, MPI_Fint *newinfo, MPI_Fint *ierr));
PN2(void, MPI_Info_free, mpi_info_free, MPI_INFO_FREE, (MPI_Fint *info, MPI_Fint *ierr));
PN2(void, MPI_Info_get, mpi_info_get, MPI_INFO_GET, (MPI_Fint *info, char *key, MPI_Fint *valuelen, char *value, ompi_fortran_logical_t *flag, MPI_Fint *ierr, int key_len, int value_len));
PN2(void, MPI_Info_get_nkeys, mpi_info_get_nkeys, MPI_INFO_GET_NKEYS, (MPI_Fint *info, MPI_Fint *nkeys, MPI_Fint *ierr));
PN2(void, MPI_Info_get_nthkey, mpi_info_get_nthkey, MPI_INFO_GET_NTHKEY, (MPI_Fint *info, MPI_Fint *n, char *key, MPI_Fint *ierr, int key_len));
PN2(void, MPI_Info_get_valuelen, mpi_info_get_valuelen, MPI_INFO_GET_VALUELEN, (MPI_Fint *info, char *key, MPI_Fint *valuelen, ompi_fortran_logical_t *flag, MPI_Fint *ierr, int key_len));
PN2(void, MPI_Info_set, mpi_info_set, MPI_INFO_SET, (MPI_Fint *info, char *key, char *value, MPI_Fint *ierr, int key_len, int value_len));
PN2(void, MPI_Init, mpi_init, MPI_INIT, (MPI_Fint *ierr));
PN2(void, MPI_Initialized, mpi_initialized, MPI_INITIALIZED, (ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Init_thread, mpi_init_thread, MPI_INIT_THREAD, (MPI_Fint *required, MPI_Fint *provided, MPI_Fint *ierr));
PN2(void, MPI_Intercomm_create, mpi_intercomm_create, MPI_INTERCOMM_CREATE, (MPI_Fint *local_comm, MPI_Fint *local_leader, MPI_Fint *bridge_comm, MPI_Fint *remote_leader, MPI_Fint *tag, MPI_Fint *newintercomm, MPI_Fint *ierr));
PN2(void, MPI_Intercomm_merge, mpi_intercomm_merge, MPI_INTERCOMM_MERGE, (MPI_Fint *intercomm, ompi_fortran_logical_t *high, MPI_Fint *newintercomm, MPI_Fint *ierr));
PN2(void, MPI_Iprobe, mpi_iprobe, MPI_IPROBE, (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, ompi_fortran_logical_t *flag, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Irecv, mpi_irecv, MPI_IRECV, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Irsend, mpi_irsend, MPI_IRSEND, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Isend, mpi_isend, MPI_ISEND, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Issend, mpi_issend, MPI_ISSEND, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Is_thread_main, mpi_is_thread_main, MPI_IS_THREAD_MAIN, (ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Keyval_create, mpi_keyval_create, MPI_KEYVAL_CREATE, (ompi_fint_copy_attr_function* copy_fn, ompi_fint_delete_attr_function* delete_fn, MPI_Fint *keyval, MPI_Fint *extra_state, MPI_Fint *ierr));
PN2(void, MPI_Keyval_free, mpi_keyval_free, MPI_KEYVAL_FREE, (MPI_Fint *keyval, MPI_Fint *ierr));
PN2(void, MPI_Lookup_name, mpi_lookup_name, MPI_LOOKUP_NAME, (char *service_name, MPI_Fint *info, char *port_name, MPI_Fint *ierr, int service_name_len, int port_name_len));
PN2(void, MPI_Mprobe, mpi_mprobe, MPI_MPROBE, (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *message, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Mrecv, mpi_mrecv, MPI_MRECV, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *message, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Neighbor_allgather, mpi_neighbor_allgather, MPI_NEIGHBOR_ALLGATHER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Neighbor_allgatherv, mpi_neighbor_allgatherv, MPI_NEIGHBOR_ALLGATHERV, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *displs, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Neighbor_alltoall, mpi_neighbor_alltoall, MPI_NEIGHBOR_ALLTOALL, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Neighbor_alltoallv, mpi_neighbor_alltoallv, MPI_NEIGHBOR_ALLTOALLV, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *sdispls, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *rdispls, MPI_Fint *recvtype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Neighbor_alltoallw, mpi_neighbor_alltoallw, MPI_NEIGHBOR_ALLTOALLW, (char *sendbuf, MPI_Fint *sendcounts, MPI_Aint *sdispls, MPI_Fint *sendtypes, char *recvbuf, MPI_Fint *recvcounts, MPI_Aint *rdispls, MPI_Fint *recvtypes, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Op_commutative, mpi_op_commutative, MPI_OP_COMMUTATIVE, (MPI_Fint *op, MPI_Fint *commute, MPI_Fint *ierr));
PN2(void, MPI_Op_create, mpi_op_create, MPI_OP_CREATE, (ompi_op_fortran_handler_fn_t* function, ompi_fortran_logical_t *commute, MPI_Fint *op, MPI_Fint *ierr));
PN2(void, MPI_Open_port, mpi_open_port, MPI_OPEN_PORT, (MPI_Fint *info, char *port_name, MPI_Fint *ierr, int port_name_len));
PN2(void, MPI_Op_free, mpi_op_free, MPI_OP_FREE, (MPI_Fint *op, MPI_Fint *ierr));
PN2(void, MPI_Pack_external, mpi_pack_external, MPI_PACK_EXTERNAL, (char *datarep, char *inbuf, MPI_Fint *incount, MPI_Fint *datatype, char *outbuf, MPI_Aint *outsize, MPI_Aint *position, MPI_Fint *ierr, int datarep_len));
PN2(void, MPI_Pack_external_size, mpi_pack_external_size, MPI_PACK_EXTERNAL_SIZE, (char *datarep, MPI_Fint *incount, MPI_Fint *datatype, MPI_Aint *size, MPI_Fint *ierr, int datarep_len));
PN2(void, MPI_Pack, mpi_pack, MPI_PACK, (char *inbuf, MPI_Fint *incount, MPI_Fint *datatype, char *outbuf, MPI_Fint *outsize, MPI_Fint *position, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Pack_size, mpi_pack_size, MPI_PACK_SIZE, (MPI_Fint *incount, MPI_Fint *datatype, MPI_Fint *comm, MPI_Fint *size, MPI_Fint *ierr));
PN2(void, MPI_Pcontrol, mpi_pcontrol, MPI_PCONTROL, (MPI_Fint *level));
PN2(void, MPI_Probe, mpi_probe, MPI_PROBE, (MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Publish_name, mpi_publish_name, MPI_PUBLISH_NAME, (char *service_name, MPI_Fint *info, char *port_name, MPI_Fint *ierr, int service_name_len, int port_name_len));
PN2(void, MPI_Put, mpi_put, MPI_PUT, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Query_thread, mpi_query_thread, MPI_QUERY_THREAD, (MPI_Fint *provided, MPI_Fint *ierr));
PN2(void, MPI_Raccumulate, mpi_raccumulate, MPI_RACCUMULATE, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Recv_init, mpi_recv_init, MPI_RECV_INIT, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Recv, mpi_recv, MPI_RECV, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *source, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Reduce, mpi_reduce, MPI_REDUCE, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Reduce_local, mpi_reduce_local, MPI_REDUCE_LOCAL, (char *inbuf, char *inoutbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *ierr));
PN2(void, MPI_Reduce_scatter, mpi_reduce_scatter, MPI_REDUCE_SCATTER, (char *sendbuf, char *recvbuf, MPI_Fint *recvcounts, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Reduce_scatter_block, mpi_reduce_scatter_block, MPI_REDUCE_SCATTER_BLOCK, (char *sendbuf, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Register_datarep, mpi_register_datarep, MPI_REGISTER_DATAREP, (char *datarep, ompi_mpi2_fortran_datarep_conversion_fn_t *read_conversion_fn, ompi_mpi2_fortran_datarep_conversion_fn_t *write_conversion_fn, ompi_mpi2_fortran_datarep_extent_fn_t *dtype_file_extent_fn, MPI_Aint *extra_state, MPI_Fint *ierr, int datarep_len));
PN2(void, MPI_Request_free, mpi_request_free, MPI_REQUEST_FREE, (MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Request_get_status, mpi_request_get_status, MPI_REQUEST_GET_STATUS, (MPI_Fint *request, ompi_fortran_logical_t *flag, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Rget, mpi_rget, MPI_RGET, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Rget_accumulate, mpi_rget_accumulate, MPI_RGET_ACCUMULATE, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, char *result_addr, MPI_Fint *result_count, MPI_Fint *result_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *op, MPI_Fint *win, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Rput, mpi_rput, MPI_RPUT, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Rsend, mpi_rsend, MPI_RSEND, (char *ibuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Rsend_init, mpi_rsend_init, MPI_RSEND_INIT, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Scan, mpi_scan, MPI_SCAN, (char *sendbuf, char *recvbuf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *op, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Scatter, mpi_scatter, MPI_SCATTER, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Scatterv, mpi_scatterv, MPI_SCATTERV, (char *sendbuf, MPI_Fint *sendcounts, MPI_Fint *displs, MPI_Fint *sendtype, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *root, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Send_init, mpi_send_init, MPI_SEND_INIT, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Send, mpi_send, MPI_SEND, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Sendrecv, mpi_sendrecv, MPI_SENDRECV, (char *sendbuf, MPI_Fint *sendcount, MPI_Fint *sendtype, MPI_Fint *dest, MPI_Fint *sendtag, char *recvbuf, MPI_Fint *recvcount, MPI_Fint *recvtype, MPI_Fint *source, MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Sendrecv_replace, mpi_sendrecv_replace, MPI_SENDRECV_REPLACE, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *sendtag, MPI_Fint *source, MPI_Fint *recvtag, MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Ssend_init, mpi_ssend_init, MPI_SSEND_INIT, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Ssend, mpi_ssend, MPI_SSEND, (char *buf, MPI_Fint *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Start, mpi_start, MPI_START, (MPI_Fint *request, MPI_Fint *ierr));
PN2(void, MPI_Startall, mpi_startall, MPI_STARTALL, (MPI_Fint *count, MPI_Fint *array_of_requests, MPI_Fint *ierr));
PN2(void, MPI_Status_set_cancelled, mpi_status_set_cancelled, MPI_STATUS_SET_CANCELLED, (MPI_Fint *status, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Status_set_elements, mpi_status_set_elements, MPI_STATUS_SET_ELEMENTS, (MPI_Fint *status, MPI_Fint *datatype, MPI_Fint *count, MPI_Fint *ierr));
PN2(void, MPI_Status_set_elements_x, mpi_status_set_elements_x, MPI_STATUS_SET_ELEMENTS_X, (MPI_Fint *status, MPI_Fint *datatype, MPI_Count *count, MPI_Fint *ierr));
PN2(void, MPI_Testall, mpi_testall, MPI_TESTALL, (MPI_Fint *count, MPI_Fint *array_of_requests, ompi_fortran_logical_t *flag, MPI_Fint *array_of_statuses, MPI_Fint *ierr));
PN2(void, MPI_Testany, mpi_testany, MPI_TESTANY, (MPI_Fint *count, MPI_Fint *array_of_requests, MPI_Fint *index, ompi_fortran_logical_t *flag, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Test, mpi_test, MPI_TEST, (MPI_Fint *request, ompi_fortran_logical_t *flag, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Test_cancelled, mpi_test_cancelled, MPI_TEST_CANCELLED, (MPI_Fint *status, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Testsome, mpi_testsome, MPI_TESTSOME, (MPI_Fint *incount, MPI_Fint *array_of_requests, MPI_Fint *outcount, MPI_Fint *array_of_indices, MPI_Fint *array_of_statuses, MPI_Fint *ierr));
PN2(void, MPI_Topo_test, mpi_topo_test, MPI_TOPO_TEST, (MPI_Fint *comm, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Type_commit, mpi_type_commit, MPI_TYPE_COMMIT, (MPI_Fint *type, MPI_Fint *ierr));
PN2(void, MPI_Type_contiguous, mpi_type_contiguous, MPI_TYPE_CONTIGUOUS, (MPI_Fint *count, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_darray, mpi_type_create_darray, MPI_TYPE_CREATE_DARRAY, (MPI_Fint *size, MPI_Fint *rank, MPI_Fint *ndims, MPI_Fint *gsize_array, MPI_Fint *distrib_array, MPI_Fint *darg_array, MPI_Fint *psize_array, MPI_Fint *order, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_f90_complex, mpi_type_create_f90_complex, MPI_TYPE_CREATE_F90_COMPLEX, (MPI_Fint *p, MPI_Fint *r, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_f90_integer, mpi_type_create_f90_integer, MPI_TYPE_CREATE_F90_INTEGER, (MPI_Fint *r, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_f90_real, mpi_type_create_f90_real, MPI_TYPE_CREATE_F90_REAL, (MPI_Fint *p, MPI_Fint *r, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_hindexed, mpi_type_create_hindexed, MPI_TYPE_CREATE_HINDEXED, (MPI_Fint *count, MPI_Fint *array_of_blocklengths, MPI_Aint *array_of_displacements, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_hvector, mpi_type_create_hvector, MPI_TYPE_CREATE_HVECTOR, (MPI_Fint *count, MPI_Fint *blocklength, MPI_Aint *stride, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_keyval, mpi_type_create_keyval, MPI_TYPE_CREATE_KEYVAL, (ompi_aint_copy_attr_function* type_copy_attr_fn, ompi_aint_delete_attr_function* type_delete_attr_fn, MPI_Fint *type_keyval, MPI_Aint *extra_state, MPI_Fint *ierr));
PN2(void, MPI_Type_create_indexed_block, mpi_type_create_indexed_block, MPI_TYPE_CREATE_INDEXED_BLOCK, (MPI_Fint *count, MPI_Fint *blocklength, MPI_Fint *array_of_displacements, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_hindexed_block, mpi_type_create_hindexed_block, MPI_TYPE_CREATE_HINDEXED_BLOCK, (MPI_Fint *count, MPI_Fint *blocklength, MPI_Aint *array_of_displacements, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_struct, mpi_type_create_struct, MPI_TYPE_CREATE_STRUCT, (MPI_Fint *count, MPI_Fint *array_of_block_lengths, MPI_Aint *array_of_displacements, MPI_Fint *array_of_types, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_subarray, mpi_type_create_subarray, MPI_TYPE_CREATE_SUBARRAY, (MPI_Fint *ndims, MPI_Fint *size_array, MPI_Fint *subsize_array, MPI_Fint *start_array, MPI_Fint *order, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_create_resized, mpi_type_create_resized, MPI_TYPE_CREATE_RESIZED, (MPI_Fint *oldtype, MPI_Aint *lb, MPI_Aint *extent, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_delete_attr, mpi_type_delete_attr, MPI_TYPE_DELETE_ATTR, (MPI_Fint *type, MPI_Fint *type_keyval, MPI_Fint *ierr));
PN2(void, MPI_Type_dup, mpi_type_dup, MPI_TYPE_DUP, (MPI_Fint *type, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_extent, mpi_type_extent, MPI_TYPE_EXTENT, (MPI_Fint *type, MPI_Fint *extent, MPI_Fint *ierr));
PN2(void, MPI_Type_free, mpi_type_free, MPI_TYPE_FREE, (MPI_Fint *type, MPI_Fint *ierr));
PN2(void, MPI_Type_free_keyval, mpi_type_free_keyval, MPI_TYPE_FREE_KEYVAL, (MPI_Fint *type_keyval, MPI_Fint *ierr));
PN2(void, MPI_Type_get_attr, mpi_type_get_attr, MPI_TYPE_GET_ATTR, (MPI_Fint *type, MPI_Fint *type_keyval, MPI_Aint *attribute_val, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Type_get_contents, mpi_type_get_contents, MPI_TYPE_GET_CONTENTS, (MPI_Fint *mtype, MPI_Fint *max_integers, MPI_Fint *max_addresses, MPI_Fint *max_datatypes, MPI_Fint *array_of_integers, MPI_Aint *array_of_addresses, MPI_Fint *array_of_datatypes, MPI_Fint *ierr));
PN2(void, MPI_Type_get_envelope, mpi_type_get_envelope, MPI_TYPE_GET_ENVELOPE, (MPI_Fint *type, MPI_Fint *num_integers, MPI_Fint *num_addresses, MPI_Fint *num_datatypes, MPI_Fint *combiner, MPI_Fint *ierr));
PN2(void, MPI_Type_get_extent, mpi_type_get_extent, MPI_TYPE_GET_EXTENT, (MPI_Fint *type, MPI_Aint *lb, MPI_Aint *extent, MPI_Fint *ierr));
PN2(void, MPI_Type_get_extent_x, mpi_type_get_extent_x, MPI_TYPE_GET_EXTENT_X, (MPI_Fint *type, MPI_Count *lb, MPI_Count *extent, MPI_Fint *ierr));
PN2(void, MPI_Type_get_name, mpi_type_get_name, MPI_TYPE_GET_NAME, (MPI_Fint *type, char *type_name, MPI_Fint *resultlen, MPI_Fint *ierr, int name_len));
PN2(void, MPI_Type_get_true_extent, mpi_type_get_true_extent, MPI_TYPE_GET_TRUE_EXTENT, (MPI_Fint *datatype, MPI_Aint *true_lb, MPI_Aint *true_extent, MPI_Fint *ierr));
PN2(void, MPI_Type_get_true_extent_x, mpi_type_get_true_extent_x, MPI_TYPE_GET_TRUE_EXTENT_X, (MPI_Fint *datatype, MPI_Count *true_lb, MPI_Count *true_extent, MPI_Fint *ierr));
PN2(void, MPI_Type_hindexed, mpi_type_hindexed, MPI_TYPE_HINDEXED, (MPI_Fint *count, MPI_Fint *array_of_blocklengths, MPI_Fint *array_of_displacements, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_hvector, mpi_type_hvector, MPI_TYPE_HVECTOR, (MPI_Fint *count, MPI_Fint *blocklength, MPI_Fint *stride, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_indexed, mpi_type_indexed, MPI_TYPE_INDEXED, (MPI_Fint *count, MPI_Fint *array_of_blocklengths, MPI_Fint *array_of_displacements, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_lb, mpi_type_lb, MPI_TYPE_LB, (MPI_Fint *type, MPI_Fint *lb, MPI_Fint *ierr));
PN2(void, MPI_Type_match_size, mpi_type_match_size, MPI_TYPE_MATCH_SIZE, (MPI_Fint *typeclass, MPI_Fint *size, MPI_Fint *type, MPI_Fint *ierr));
PN2(void, MPI_Type_set_attr, mpi_type_set_attr, MPI_TYPE_SET_ATTR, (MPI_Fint *type, MPI_Fint *type_keyval, MPI_Aint *attr_val, MPI_Fint *ierr));
PN2(void, MPI_Type_set_name, mpi_type_set_name, MPI_TYPE_SET_NAME, (MPI_Fint *type, char *type_name, MPI_Fint *ierr, int name_len));
PN2(void, MPI_Type_size, mpi_type_size, MPI_TYPE_SIZE, (MPI_Fint *type, MPI_Fint *size, MPI_Fint *ierr));
PN2(void, MPI_Type_size_x, mpi_type_size_x, MPI_TYPE_SIZE_X, (MPI_Fint *type, MPI_Count *size, MPI_Fint *ierr));
PN2(void, MPI_Type_struct, mpi_type_struct, MPI_TYPE_STRUCT, (MPI_Fint *count, MPI_Fint *array_of_blocklengths, MPI_Fint *array_of_displacements, MPI_Fint *array_of_types, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Type_ub, mpi_type_ub, MPI_TYPE_UB, (MPI_Fint *mtype, MPI_Fint *ub, MPI_Fint *ierr));
PN2(void, MPI_Type_vector, mpi_type_vector, MPI_TYPE_VECTOR, (MPI_Fint *count, MPI_Fint *blocklength, MPI_Fint *stride, MPI_Fint *oldtype, MPI_Fint *newtype, MPI_Fint *ierr));
PN2(void, MPI_Unpack, mpi_unpack, MPI_UNPACK, (char *inbuf, MPI_Fint *insize, MPI_Fint *position, char *outbuf, MPI_Fint *outcount, MPI_Fint *datatype, MPI_Fint *comm, MPI_Fint *ierr));
PN2(void, MPI_Unpublish_name, mpi_unpublish_name, MPI_UNPUBLISH_NAME, (char *service_name, MPI_Fint *info, char *port_name, MPI_Fint *ierr, int service_name_len, int port_name_len));
PN2(void, MPI_Unpack_external, mpi_unpack_external, MPI_UNPACK_EXTERNAL, (char *datarep, char *inbuf, MPI_Aint *insize, MPI_Aint *position, char *outbuf, MPI_Fint *outcount, MPI_Fint *datatype, MPI_Fint *ierr, int datarep_len));
PN2(void, MPI_Waitall, mpi_waitall, MPI_WAITALL, (MPI_Fint *count, MPI_Fint *array_of_requests, MPI_Fint *array_of_statuses, MPI_Fint *ierr));
PN2(void, MPI_Waitany, mpi_waitany, MPI_WAITANY, (MPI_Fint *count, MPI_Fint *array_of_requests, MPI_Fint *index, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Wait, mpi_wait, MPI_WAIT, (MPI_Fint *request, MPI_Fint *status, MPI_Fint *ierr));
PN2(void, MPI_Waitsome, mpi_waitsome, MPI_WAITSOME, (MPI_Fint *incount, MPI_Fint *array_of_requests, MPI_Fint *outcount, MPI_Fint *array_of_indices, MPI_Fint *array_of_statuses, MPI_Fint *ierr));
PN2(void, MPI_Win_allocate, mpi_win_allocate, MPI_WIN_ALLOCATE, (MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info, MPI_Fint *comm, char *baseptr, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_allocate_cptr, mpi_win_allocate_cptr, MPI_WIN_ALLOCATE_CPTR, (MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info, MPI_Fint *comm, char *baseptr, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_allocate_shared, mpi_win_allocate_shared, MPI_WIN_ALLOCATE_SHARED, (MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info, MPI_Fint *comm, char *baseptr, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_allocate_shared_cptr, mpi_win_allocate_shared_cptr, MPI_WIN_ALLOCATE_SHARED_CPTR, (MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info, MPI_Fint *comm, char *baseptr, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_attach, mpi_win_attach, MPI_WIN_ATTACH, (MPI_Fint *win, char *base, MPI_Aint *size, MPI_Fint *ierr));
PN2(void, MPI_Win_call_errhandler, mpi_win_call_errhandler, MPI_WIN_CALL_ERRHANDLER, (MPI_Fint *win, MPI_Fint *errorcode, MPI_Fint *ierr));
PN2(void, MPI_Win_complete, mpi_win_complete, MPI_WIN_COMPLETE, (MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_create, mpi_win_create, MPI_WIN_CREATE, (char *base, MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info, MPI_Fint *comm, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_create_dynamic, mpi_win_create_dynamic, MPI_WIN_CREATE_DYNAMIC, (MPI_Fint *info, MPI_Fint *comm, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_create_errhandler, mpi_win_create_errhandler, MPI_WIN_CREATE_ERRHANDLER, (ompi_errhandler_fortran_handler_fn_t* function, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Win_create_keyval, mpi_win_create_keyval, MPI_WIN_CREATE_KEYVAL, (ompi_aint_copy_attr_function* win_copy_attr_fn, ompi_aint_delete_attr_function* win_delete_attr_fn, MPI_Fint *win_keyval, MPI_Aint *extra_state, MPI_Fint *ierr));
PN2(void, MPI_Win_delete_attr, mpi_win_delete_attr, MPI_WIN_DELETE_ATTR, (MPI_Fint *win, MPI_Fint *win_keyval, MPI_Fint *ierr));
PN2(void, MPI_Win_detach, mpi_win_detach, MPI_WIN_DETACH, (MPI_Fint *win, char *base, MPI_Fint *ierr));
PN2(void, MPI_Win_fence, mpi_win_fence, MPI_WIN_FENCE, (MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_flush, mpi_win_flush, MPI_WIN_FLUSH, (MPI_Fint *rank, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_flush_all, mpi_win_flush_all, MPI_WIN_FLUSH_ALL, (MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_flush_local, mpi_win_flush_local, MPI_WIN_FLUSH_LOCAL, (MPI_Fint *rank, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_flush_local_all, mpi_win_flush_local_all, MPI_WIN_FLUSH_LOCAL_ALL, (MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_free, mpi_win_free, MPI_WIN_FREE, (MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_free_keyval, mpi_win_free_keyval, MPI_WIN_FREE_KEYVAL, (MPI_Fint *win_keyval, MPI_Fint *ierr));
PN2(void, MPI_Win_get_attr, mpi_win_get_attr, MPI_WIN_GET_ATTR, (MPI_Fint *win, MPI_Fint *win_keyval, MPI_Aint *attribute_val, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Win_get_errhandler, mpi_win_get_errhandler, MPI_WIN_GET_ERRHANDLER, (MPI_Fint *win, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Win_get_group, mpi_win_get_group, MPI_WIN_GET_GROUP, (MPI_Fint *win, MPI_Fint *group, MPI_Fint *ierr));
PN2(void, MPI_Win_get_info, mpi_win_get_info, MPI_WIN_GET_INFO, (MPI_Fint *win, MPI_Fint *info, MPI_Fint *ierr));
PN2(void, MPI_Win_get_name, mpi_win_get_name, MPI_WIN_GET_NAME, (MPI_Fint *win, char *win_name, MPI_Fint *resultlen, MPI_Fint *ierr, int name_len));
PN2(void, MPI_Win_lock, mpi_win_lock, MPI_WIN_LOCK, (MPI_Fint *lock_type, MPI_Fint *rank, MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_lock_all, mpi_win_lock_all, MPI_WIN_LOCK_ALL, (MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_post, mpi_win_post, MPI_WIN_POST, (MPI_Fint *group, MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_set_attr, mpi_win_set_attr, MPI_WIN_SET_ATTR, (MPI_Fint *win, MPI_Fint *win_keyval, MPI_Aint *attribute_val, MPI_Fint *ierr));
PN2(void, MPI_Win_set_errhandler, mpi_win_set_errhandler, MPI_WIN_SET_ERRHANDLER, (MPI_Fint *win, MPI_Fint *errhandler, MPI_Fint *ierr));
PN2(void, MPI_Win_set_info, mpi_win_set_info, MPI_WIN_SET_INFO, (MPI_Fint *win, MPI_Fint *info, MPI_Fint *ierr));
PN2(void, MPI_Win_set_name, mpi_win_set_name, MPI_WIN_SET_NAME, (MPI_Fint *win, char *win_name, MPI_Fint *ierr, int name_len));
PN2(void, MPI_Win_shared_query, mpi_win_shared_query, MPI_WIN_SHARED_QUERY, (MPI_Fint *win, MPI_Fint *rank, MPI_Aint *size, MPI_Fint *disp_unit, char *baseptr, MPI_Fint *ierr));
PN2(void, MPI_Win_shared_query_cptr, mpi_win_shared_query_cptr, MPI_WIN_SHARED_QUERY_CPTR, (MPI_Fint *win, MPI_Fint *rank, MPI_Aint *size, MPI_Fint *disp_unit, char *baseptr, MPI_Fint *ierr));
PN2(void, MPI_Win_start, mpi_win_start, MPI_WIN_START, (MPI_Fint *group, MPI_Fint *assert, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_sync, mpi_win_sync, MPI_WIN_SYNC, (MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_test, mpi_win_test, MPI_WIN_TEST, (MPI_Fint *win, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPI_Win_unlock, mpi_win_unlock, MPI_WIN_UNLOCK, (MPI_Fint *rank, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_unlock_all, mpi_win_unlock_all, MPI_WIN_UNLOCK_ALL, (MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPI_Win_wait, mpi_win_wait, MPI_WIN_WAIT, (MPI_Fint *win, MPI_Fint *ierr));
PN2(double, MPI_Wtick, mpi_wtick, MPI_WTICK, (void));
PN2(double, MPI_Wtime, mpi_wtime, MPI_WTIME, (void));

PN2(void, MPI_Type_null_delete_fn, mpi_type_null_delete_fn, MPI_TYPE_NULL_DELETE_FN, (MPI_Fint* type, MPI_Fint* type_keyval, MPI_Aint* attribute_val_out, MPI_Aint* extra_state, MPI_Fint* ierr));
PN2(void, MPI_Type_null_copy_fn, mpi_type_null_copy_fn, MPI_TYPE_NULL_COPY_FN, (MPI_Fint* type, MPI_Fint* type_keyval, MPI_Aint* extra_state, MPI_Aint* attribute_val_in, MPI_Aint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));
PN2(void, MPI_Type_dup_fn, mpi_type_dup_fn, MPI_TYPE_DUP_FN, (MPI_Fint* type, MPI_Fint* type_keyval, MPI_Aint* extra_state, MPI_Aint* attribute_val_in, MPI_Aint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));
PN2(void, MPI_Win_dup_fn, mpi_win_dup_fn, MPI_WIN_DUP_FN, (MPI_Fint* window, MPI_Fint* win_keyval, MPI_Aint* extra_state, MPI_Aint* attribute_val_in, MPI_Aint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));
PN2(void, MPI_Win_null_copy_fn, mpi_win_null_copy_fn, MPI_WIN_NULL_COPY_FN, (MPI_Fint* window, MPI_Fint* win_keyval, MPI_Aint* extra_state, MPI_Aint* attribute_val_in, MPI_Aint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));
PN2(void, MPI_Win_null_delete_fn, mpi_win_null_delete_fn, MPI_WIN_NULL_DELETE_FN, (MPI_Fint* window, MPI_Fint* win_keyval, MPI_Aint* attribute_val_out, MPI_Aint* extra_state, MPI_Fint* ierr));
PN2(void, MPI_Null_delete_fn, mpi_null_delete_fn, MPI_NULL_DELETE_FN, (MPI_Fint* comm, MPI_Fint* comm_keyval, MPI_Fint* attribute_val_out, MPI_Fint* extra_state, MPI_Fint* ierr));
PN2(void, MPI_Null_copy_fn, mpi_null_copy_fn, MPI_NULL_COPY_FN, (MPI_Fint* comm, MPI_Fint* comm_keyval, MPI_Fint* extra_state, MPI_Fint* attribute_val_in, MPI_Fint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));
PN2(void, MPI_Dup_fn, mpi_dup_fn, MPI_DUP_FN, (MPI_Fint* comm, MPI_Fint* comm_keyval, MPI_Fint* extra_state, MPI_Fint* attribute_val_in, MPI_Fint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));
PN2(void, MPI_Comm_null_delete_fn, mpi_comm_null_delete_fn, MPI_COMM_NULL_DELETE_FN, (MPI_Fint* comm, MPI_Fint* comm_keyval, MPI_Aint* attribute_val_out, MPI_Aint* extra_state, MPI_Fint* ierr));
PN2(void, MPI_Comm_null_copy_fn, mpi_comm_null_copy_fn, MPI_COMM_NULL_COPY_FN, (MPI_Fint* comm, MPI_Fint* comm_keyval, MPI_Aint* extra_state, MPI_Aint* attribute_val_in, MPI_Aint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));
PN2(void, MPI_Comm_dup_fn, mpi_comm_dup_fn, MPI_COMM_DUP_FN, (MPI_Fint* comm, MPI_Fint* comm_keyval, MPI_Aint* extra_state, MPI_Aint* attribute_val_in, MPI_Aint* attribute_val_out, ompi_fortran_logical_t * flag, MPI_Fint* ierr));

END_C_DECLS

#endif
