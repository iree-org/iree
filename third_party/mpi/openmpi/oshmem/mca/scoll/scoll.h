/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * Collective Communication Interface
 *
 */

#ifndef OSHMEM_MCA_SCOLL_H
#define OSHMEM_MCA_SCOLL_H

#include "oshmem_config.h"
#include "oshmem/types.h"
#include "oshmem/constants.h"

#include "opal/util/output.h"
#include "mpi.h"
#include "oshmem/mca/mca.h"
#include "opal/mca/base/base.h"

BEGIN_C_DECLS

/* ******************************************************************** */

struct oshmem_group_t;
struct oshmem_op_t;

/* ******************************************************************** */

typedef int (*mca_scoll_base_component_init_fn_t)(bool enable_progress_threads,
                                                  bool enable_threads);

typedef struct mca_scoll_base_module_1_0_0_t* (*mca_scoll_base_component_query_fn_t)(struct oshmem_group_t *group,
                                                                                     int *priority);

/* ******************************************************************** */

/**
 * Collective component interface
 *
 * Component interface for the collective framework.  A public
 * instance of this structure, called
 * mca_scoll_[component_name]_component, must exist in any collective
 * component.
 */
struct mca_scoll_base_component_1_0_0_t {
    /** Base component description */
    mca_base_component_t scoll_version;
    /** Base component data block */
    mca_base_component_data_t scoll_data;

    /** Component initialization function */
    mca_scoll_base_component_init_fn_t scoll_init;
    mca_scoll_base_component_query_fn_t scoll_query;
};
typedef struct mca_scoll_base_component_1_0_0_t mca_scoll_base_component_1_0_0_t;

/** Per guidence in mca.h, use the unversioned struct name if you just
 want to always keep up with the most recent version of the
 interace. */
typedef struct mca_scoll_base_component_1_0_0_t mca_scoll_base_component_t;

/**
 * Collective module interface
 *
 * Module interface to the Collective framework.  Modules are
 * reference counted based on the number of functions from the module
 * used on the commuicator.  There is at most one module per component
 * on a given communicator, and there can be many component modules on
 * a given communicator.
 *
 * @note The collective framework and the
 * communicator functionality only stores a pointer to the module
 * function, so the component is free to create a structure that
 * inherits from this one for use as the module structure.
 */
typedef int
(*mca_scoll_base_module_enable_1_0_0_fn_t)(struct mca_scoll_base_module_1_0_0_t* module,
                                           struct oshmem_group_t *comm);

#define SCOLL_DEFAULT_ALG   (-1)

#define SCOLL_ALG_BARRIER_CENTRAL_COUNTER       0
#define SCOLL_ALG_BARRIER_TOURNAMENT            1
#define SCOLL_ALG_BARRIER_RECURSIVE_DOUBLING    2
#define SCOLL_ALG_BARRIER_DISSEMINATION         3
#define SCOLL_ALG_BARRIER_BASIC                 4
#define SCOLL_ALG_BARRIER_ADAPTIVE              5

#define SCOLL_ALG_BROADCAST_CENTRAL_COUNTER     0
#define SCOLL_ALG_BROADCAST_BINOMIAL            1

#define SCOLL_ALG_COLLECT_CENTRAL_COUNTER       0
#define SCOLL_ALG_COLLECT_TOURNAMENT            1
#define SCOLL_ALG_COLLECT_RECURSIVE_DOUBLING    2
#define SCOLL_ALG_COLLECT_RING                  3

#define SCOLL_ALG_REDUCE_CENTRAL_COUNTER        0
#define SCOLL_ALG_REDUCE_TOURNAMENT             1
#define SCOLL_ALG_REDUCE_RECURSIVE_DOUBLING     2
#define SCOLL_ALG_REDUCE_LEGACY_LINEAR          3   /* Based linear algorithm from OMPI coll:basic */
#define SCOLL_ALG_REDUCE_LEGACY_LOG             4   /* Based log algorithm from OMPI coll:basic */

typedef int (*mca_scoll_base_module_barrier_fn_t)(struct oshmem_group_t *group,
                                                  long *pSync,
                                                  int alg);
typedef int (*mca_scoll_base_module_broadcast_fn_t)(struct oshmem_group_t *group,
                                                    int PE_root,
                                                    void *target,
                                                    const void *source,
                                                    size_t nlong,
                                                    long *pSync,
                                                    bool nlong_type,
                                                    int alg);
typedef int (*mca_scoll_base_module_collect_fn_t)(struct oshmem_group_t *group,
                                                  void *target,
                                                  const void *source,
                                                  size_t nlong,
                                                  long *pSync,
                                                  bool nlong_type,
                                                  int alg);
typedef int (*mca_scoll_base_module_reduce_fn_t)(struct oshmem_group_t *group,
                                                 struct oshmem_op_t *op,
                                                 void *target,
                                                 const void *source,
                                                 size_t nlong,
                                                 long *pSync,
                                                 void *pWrk,
                                                 int alg);
typedef int (*mca_scoll_base_module_alltoall_fn_t)(struct oshmem_group_t *group,
                                                  void *target,
                                                  const void *source,
                                                  ptrdiff_t dst, ptrdiff_t sst,
                                                  size_t nelems,
                                                  size_t element_size,
                                                  long *pSync,
                                                  int alg);

struct mca_scoll_base_module_1_0_0_t {
    /** Collective modules all inherit from opal_object */
    opal_object_t super;

    /* Collective function pointers */
    mca_scoll_base_module_barrier_fn_t scoll_barrier;
    mca_scoll_base_module_broadcast_fn_t scoll_broadcast;
    mca_scoll_base_module_collect_fn_t scoll_collect;
    mca_scoll_base_module_reduce_fn_t scoll_reduce;
    mca_scoll_base_module_alltoall_fn_t scoll_alltoall;
    mca_scoll_base_module_enable_1_0_0_fn_t scoll_module_enable;
};
typedef struct mca_scoll_base_module_1_0_0_t mca_scoll_base_module_1_0_0_t;

/** Per guidance in mca.h, use the unversioned struct name if you just
 want to always keep up with the most recent version of the
 interface. */
typedef struct mca_scoll_base_module_1_0_0_t mca_scoll_base_module_t;
OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(mca_scoll_base_module_t);

/* ******************************************************************** */

/*
 * Macro for use in components that are of type coll
 */
#define MCA_SCOLL_BASE_VERSION_2_0_0 \
    OSHMEM_MCA_BASE_VERSION_2_1_0("scoll", 1, 0, 0)

/* ******************************************************************** */
/*
 * Collectives group cache structure
 *
 * Collectives group cache structure, used to find functions to
 * implement collective algorithms and their associated modules.
 */
struct mca_scoll_base_group_scoll_t {
    mca_scoll_base_module_barrier_fn_t scoll_barrier;
    mca_scoll_base_module_1_0_0_t *scoll_barrier_module;
    mca_scoll_base_module_broadcast_fn_t scoll_broadcast;
    mca_scoll_base_module_1_0_0_t *scoll_broadcast_module;
    mca_scoll_base_module_collect_fn_t scoll_collect;
    mca_scoll_base_module_1_0_0_t *scoll_collect_module;
    mca_scoll_base_module_reduce_fn_t scoll_reduce;
    mca_scoll_base_module_1_0_0_t *scoll_reduce_module;
    mca_scoll_base_module_alltoall_fn_t scoll_alltoall;
    mca_scoll_base_module_1_0_0_t *scoll_alltoall_module;
};
typedef struct mca_scoll_base_group_scoll_t mca_scoll_base_group_scoll_t;

#define PREVIOUS_SCOLL_FN(module, __api, group, ...) do { \
    group->g_scoll.scoll_ ## __api ## _module = (mca_scoll_base_module_1_0_0_t*) module->previous_ ## __api ## _module; \
    rc = module->previous_ ## __api (group, __VA_ARGS__); \
    group->g_scoll.scoll_ ## __api ## _module = (mca_scoll_base_module_1_0_0_t*) module; \
} while(0)

END_C_DECLS

#endif /* OSHMEM_MCA_SCOLL_H */
