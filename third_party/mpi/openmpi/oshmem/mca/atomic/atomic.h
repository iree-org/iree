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
 * Atomic Operations Interface
 *
 */

#ifndef OSHMEM_MCA_ATOMIC_H
#define OSHMEM_MCA_ATOMIC_H

#include "oshmem_config.h"
#include "oshmem/types.h"
#include "oshmem/constants.h"

#include "opal/util/output.h"
#include "mpi.h"
#include "oshmem/mca/mca.h"
#include "opal/mca/base/base.h"
#include "oshmem/mca/atomic/base/base.h"

BEGIN_C_DECLS

#define OSHMEM_ATOMIC_PTR_2_INT(ptr, size) ((size) == 8 ? *(uint64_t*)(ptr) : *(uint32_t*)(ptr))

#define DO_SHMEM_TYPE_OP(ctx, type_name, type, op, target, value, pe) do {    \
        int rc = OSHMEM_SUCCESS;                                              \
        size_t size = 0;                                                      \
                                                                              \
        RUNTIME_CHECK_INIT();                                                 \
        RUNTIME_CHECK_PE(pe);                                                 \
        RUNTIME_CHECK_ADDR(target);                                           \
                                                                              \
        size = sizeof(value);                                                 \
        rc = MCA_ATOMIC_CALL(op(                                              \
            ctx,                                                              \
            (void*)target,                                                    \
            value,                                                            \
            size,                                                             \
            pe));                                                             \
        RUNTIME_CHECK_RC(rc);                                                 \
                                                                              \
        return;                                                               \
    } while (0)

#define OSHMEM_TYPE_OP(type_name, type, prefix, op)                           \
    void prefix##_##type_name##_atomic_##op(type *target, type value, int pe) \
    {                                                                         \
        DO_SHMEM_TYPE_OP(oshmem_ctx_default, type_name, type, op,             \
                         target, value, pe);                                  \
    }

#define OSHMEM_CTX_TYPE_OP(type_name, type, prefix, op)                       \
    void prefix##_ctx_##type_name##_atomic_##op(shmem_ctx_t ctx, type *target, type value, int pe) \
    {                                                                         \
        DO_SHMEM_TYPE_OP(ctx, type_name, type, op,                            \
                         target, value, pe);                                  \
    }

#define DO_OSHMEM_TYPE_FOP(ctx, type_name, type, op, target, value, pe) do {        \
        int rc = OSHMEM_SUCCESS;                                                    \
        size_t size = 0;                                                            \
        type out_value;                                                             \
                                                                                    \
        RUNTIME_CHECK_INIT();                                                       \
        RUNTIME_CHECK_PE(pe);                                                       \
        RUNTIME_CHECK_ADDR(target);                                                 \
                                                                                    \
        size = sizeof(out_value);                                                   \
        rc = MCA_ATOMIC_CALL(f##op(                                                 \
            ctx,                                                                    \
            (void*)target,                                                          \
            (void*)&out_value,                                                      \
            value,                                                                  \
            size,                                                                   \
            pe));                                                                   \
        RUNTIME_CHECK_RC(rc);                                                       \
                                                                                    \
        return out_value;                                                           \
    } while (0)

#define OSHMEM_TYPE_FOP(type_name, type, prefix, op)                                \
    type prefix##_##type_name##_atomic_fetch_##op(type *target, type value, int pe) \
    {                                                                               \
        DO_OSHMEM_TYPE_FOP(oshmem_ctx_default, type_name, type, op,                 \
                           target, value, pe);                                      \
    }

#define OSHMEM_CTX_TYPE_FOP(type_name, type, prefix, op)                   \
    type prefix##_ctx_##type_name##_atomic_fetch_##op(shmem_ctx_t ctx, type *target, type value, int pe) \
    {                                                                      \
        DO_OSHMEM_TYPE_FOP(ctx, type_name, type, op,                       \
                           target, value, pe);                             \
    }

/* ******************************************************************** */

struct oshmem_op_t;

/* ******************************************************************** */

typedef int (*mca_atomic_base_component_init_fn_t)(bool enable_progress_threads,
                                                   bool enable_threads);

typedef int (*mca_atomic_base_component_finalize_fn_t)(void);

typedef struct mca_atomic_base_module_1_0_0_t* (*mca_atomic_base_component_query_fn_t)(int *priority);

/* ******************************************************************** */

/**
 * Atomic component interface
 *
 * Component interface for the atomic framework.  A public
 * instance of this structure, called
 * mca_atomic_[component_name]_component, must exist in any atomic
 * component.
 */
struct mca_atomic_base_component_1_0_0_t {
    /** Base component description */
    mca_base_component_t atomic_version;
    /** Base component data block */
    mca_base_component_data_t atomic_data;

    /** Component initialization function */
    mca_atomic_base_component_init_fn_t atomic_startup;
    mca_atomic_base_component_finalize_fn_t atomic_finalize;
    mca_atomic_base_component_query_fn_t atomic_query;

    /* priority for component */
    int priority;
};
typedef struct mca_atomic_base_component_1_0_0_t mca_atomic_base_component_1_0_0_t;

/** Per guidence in mca.h, use the unversioned struct name if you just
 want to always keep up with the most recent version of the
 interace. */
typedef struct mca_atomic_base_component_1_0_0_t mca_atomic_base_component_t;

/**
 * Atomic module interface
 *
 */
struct mca_atomic_base_module_1_0_0_t {
    /** Collective modules all inherit from opal_object */
    opal_object_t super;

    /* Collective function pointers */
    int (*atomic_add)(shmem_ctx_t ctx,
                      void *target,
                      uint64_t value,
                      size_t size,
                      int pe);
    int (*atomic_and)(shmem_ctx_t ctx,
                      void *target,
                      uint64_t value,
                      size_t size,
                      int pe);
    int (*atomic_or)(shmem_ctx_t ctx,
                     void *target,
                     uint64_t value,
                     size_t size,
                     int pe);
    int (*atomic_xor)(shmem_ctx_t ctx,
                      void *target,
                      uint64_t value,
                      size_t size,
                      int pe);
    int (*atomic_fadd)(shmem_ctx_t ctx,
                       void *target,
                       void *prev,
                       uint64_t value,
                       size_t size,
                       int pe);
    int (*atomic_fand)(shmem_ctx_t ctx,
                       void *target,
                       void *prev,
                       uint64_t value,
                       size_t size,
                       int pe);
    int (*atomic_for)(shmem_ctx_t ctx,
                      void *target,
                      void *prev,
                      uint64_t value,
                      size_t size,
                      int pe);
    int (*atomic_fxor)(shmem_ctx_t ctx,
                       void *target,
                       void *prev,
                       uint64_t value,
                       size_t size,
                       int pe);
    int (*atomic_swap)(shmem_ctx_t ctx,
                       void *target,
                       void *prev,
                       uint64_t value,
                       size_t size,
                       int pe);
    int (*atomic_cswap)(shmem_ctx_t ctx,
                        void *target,
                        uint64_t *prev, /* prev is used internally by wrapper, we may
                                           always use 64-bit value */
                        uint64_t cond,
                        uint64_t value,
                        size_t size,
                        int pe);
};
typedef struct mca_atomic_base_module_1_0_0_t mca_atomic_base_module_1_0_0_t;

/** Per guidence in mca.h, use the unversioned struct name if you just
 want to always keep up with the most recent version of the
 interace. */
typedef struct mca_atomic_base_module_1_0_0_t mca_atomic_base_module_t;
OSHMEM_DECLSPEC OBJ_CLASS_DECLARATION(mca_atomic_base_module_t);

/* ******************************************************************** */

/*
 * Macro for use in components
 */
#define MCA_ATOMIC_BASE_VERSION_2_0_0 \
    OSHMEM_MCA_BASE_VERSION_2_1_0("atomic", 1, 0, 0)

/* ******************************************************************** */

OSHMEM_DECLSPEC extern mca_atomic_base_component_t mca_atomic_base_selected_component;
OSHMEM_DECLSPEC extern mca_atomic_base_module_t mca_atomic;
#define MCA_ATOMIC_CALL(a) mca_atomic.atomic_ ## a

END_C_DECLS

#endif /* OSHMEM_MCA_ATOMIC_H */
