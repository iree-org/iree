/*
 * Copyright (c) 2018      Mellanox Technologies.  All rights reserved.
 *                         All rights reserved.
 * Copyright (c) 2022      IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef _COMMON_UCX_H_
#define _COMMON_UCX_H_

#include "opal_config.h"

#include <stdint.h>

#include <ucp/api/ucp.h>

#include "opal/mca/mca.h"
#include "opal/util/output.h"
#include "opal/runtime/opal_progress.h"
#include "opal/include/opal/constants.h"
#include "opal/class/opal_list.h"

BEGIN_C_DECLS

#define MCA_COMMON_UCX_ENABLE_DEBUG   OPAL_ENABLE_DEBUG
#if MCA_COMMON_UCX_ENABLE_DEBUG
#  define MCA_COMMON_UCX_MAX_VERBOSE  100
#  define MCA_COMMON_UCX_ASSERT(_x)   assert(_x)
#else
#  define MCA_COMMON_UCX_MAX_VERBOSE  2
#  define MCA_COMMON_UCX_ASSERT(_x)
#endif

#define UCX_VERSION(_major, _minor, _build) (((_major) * 100) + (_minor))

#define _MCA_COMMON_UCX_QUOTE(_x) \
    # _x
#define MCA_COMMON_UCX_QUOTE(_x) \
    _MCA_COMMON_UCX_QUOTE(_x)

#define MCA_COMMON_UCX_ERROR(...) \
    MCA_COMMON_UCX_VERBOSE(0, " Error: " __VA_ARGS__)

#define MCA_COMMON_UCX_WARN(...) \
    MCA_COMMON_UCX_VERBOSE(0, " Warning: " __VA_ARGS__)

#define MCA_COMMON_UCX_VERBOSE(_level, ... )                                \
    if (((_level) <= MCA_COMMON_UCX_MAX_VERBOSE) &&                         \
        ((_level) <= opal_common_ucx.verbose)) {                            \
        opal_output_verbose(_level, opal_common_ucx.output,                 \
                            __FILE__ ":" MCA_COMMON_UCX_QUOTE(__LINE__) " " \
                            __VA_ARGS__);                                   \
    }

/* progress loop to allow call UCX/opal progress */
/* used C99 for-statement variable initialization */
#define MCA_COMMON_UCX_PROGRESS_LOOP(_worker)                                 \
    for (unsigned iter = 0;; (++iter % opal_common_ucx.progress_iterations) ? \
                        (void)ucp_worker_progress(_worker) : opal_progress())

#define MCA_COMMON_UCX_WAIT_LOOP(_request, _worker, _msg, _completed)                    \
    do {                                                                                 \
        ucs_status_t status;                                                             \
        /* call UCX progress */                                                          \
        MCA_COMMON_UCX_PROGRESS_LOOP(_worker) {                                          \
            status = opal_common_ucx_request_status(_request);                           \
            if (UCS_INPROGRESS != status) {                                              \
                _completed;                                                              \
                if (OPAL_LIKELY(UCS_OK == status)) {                                     \
                    return OPAL_SUCCESS;                                                 \
                } else {                                                                 \
                    MCA_COMMON_UCX_VERBOSE(1, "%s failed: %d, %s",                       \
                                           (_msg) ? (_msg) : __func__,                   \
                                           UCS_PTR_STATUS(_request),                     \
                                           ucs_status_string(UCS_PTR_STATUS(_request))); \
                    return OPAL_ERROR;                                                   \
                }                                                                        \
            }                                                                            \
        }                                                                                \
    } while (0)

typedef struct opal_common_ucx_module {
    int  output;
    int  verbose;
    int  progress_iterations;
    int  registered;
    bool opal_mem_hooks;
    char **tls;
    char **devices;
} opal_common_ucx_module_t;

typedef struct opal_common_ucx_del_proc {
    ucp_ep_h ep;
    size_t   vpid;
} opal_common_ucx_del_proc_t;

typedef enum {
    /* No supported transports found (according to configured list of supported
       transports) */
    OPAL_COMMON_UCX_SUPPORT_NONE,

    /* Have supported transports but not supported devices */
    OPAL_COMMON_UCX_SUPPORT_TRANSPORT,

    /* Have both supported transports and supported devices */
    OPAL_COMMON_UCX_SUPPORT_DEVICE,
} opal_common_ucx_support_level_t;

extern opal_common_ucx_module_t opal_common_ucx;

OPAL_DECLSPEC void opal_common_ucx_mca_register(void);
OPAL_DECLSPEC void opal_common_ucx_mca_deregister(void);
OPAL_DECLSPEC opal_common_ucx_support_level_t opal_common_ucx_support_level(ucp_context_h context);
OPAL_DECLSPEC void opal_common_ucx_mca_proc_added(void);
OPAL_DECLSPEC void opal_common_ucx_empty_complete_cb(void *request, ucs_status_t status);
OPAL_DECLSPEC int opal_common_ucx_mca_pmix_fence(ucp_worker_h worker);
OPAL_DECLSPEC int opal_common_ucx_mca_pmix_fence_nb(int *fenced);
OPAL_DECLSPEC int opal_common_ucx_del_procs(opal_common_ucx_del_proc_t *procs, size_t count,
                                            size_t my_rank, size_t max_disconnect, ucp_worker_h worker);
OPAL_DECLSPEC int opal_common_ucx_del_procs_nofence(opal_common_ucx_del_proc_t *procs, size_t count,
                                               size_t my_rank, size_t max_disconnect, ucp_worker_h worker);
OPAL_DECLSPEC void opal_common_ucx_mca_var_register(const mca_base_component_t *component);

static inline
ucs_status_t opal_common_ucx_request_status(ucs_status_ptr_t request)
{
#if !HAVE_DECL_UCP_REQUEST_CHECK_STATUS
    ucp_tag_recv_info_t info;

    return ucp_request_test(request, &info);
#else
    return ucp_request_check_status(request);
#endif
}

static inline
int opal_common_ucx_wait_request(ucs_status_ptr_t request, ucp_worker_h worker,
                                 const char *msg)
{
    /* check for request completed or failed */
    if (OPAL_LIKELY(UCS_OK == request)) {
        return OPAL_SUCCESS;
    } else if (OPAL_UNLIKELY(UCS_PTR_IS_ERR(request))) {
        MCA_COMMON_UCX_VERBOSE(1, "%s failed: %d, %s", msg ? msg : __func__,
                               UCS_PTR_STATUS(request),
                               ucs_status_string(UCS_PTR_STATUS(request)));
        return OPAL_ERROR;
    }

    MCA_COMMON_UCX_WAIT_LOOP(request, worker, msg, ucp_request_free(request));
}

static inline
int opal_common_ucx_ep_flush(ucp_ep_h ep, ucp_worker_h worker)
{
#if HAVE_DECL_UCP_EP_FLUSH_NB
    ucs_status_ptr_t request;

    request = ucp_ep_flush_nb(ep, 0, opal_common_ucx_empty_complete_cb);
    return opal_common_ucx_wait_request(request, worker, "ucp_ep_flush_nb");
#else
    ucs_status_t status;

    status = ucp_ep_flush(ep);
    return (status == UCS_OK) ? OPAL_SUCCESS : OPAL_ERROR;
#endif
}

static inline
int opal_common_ucx_worker_flush(ucp_worker_h worker)
{
#if HAVE_DECL_UCP_WORKER_FLUSH_NB
    ucs_status_ptr_t request;

    request = ucp_worker_flush_nb(worker, 0, opal_common_ucx_empty_complete_cb);
    return opal_common_ucx_wait_request(request, worker, "ucp_worker_flush_nb");
#else
    ucs_status_t status;

    status = ucp_worker_flush(worker);
    return (status == UCS_OK) ? OPAL_SUCCESS : OPAL_ERROR;
#endif
}

static inline
int opal_common_ucx_atomic_fetch(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                                 uint64_t value, void *result, size_t op_size,
                                 uint64_t remote_addr, ucp_rkey_h rkey,
                                 ucp_worker_h worker)
{
    ucs_status_ptr_t request;

    request = ucp_atomic_fetch_nb(ep, opcode, value, result, op_size,
                                  remote_addr, rkey, opal_common_ucx_empty_complete_cb);
    return opal_common_ucx_wait_request(request, worker, "ucp_atomic_fetch_nb");
}

static inline
int opal_common_ucx_atomic_cswap(ucp_ep_h ep, uint64_t compare,
                                 uint64_t value, void *result, size_t op_size,
                                 uint64_t remote_addr, ucp_rkey_h rkey,
                                 ucp_worker_h worker)
{
    uint64_t tmp = value;
    int ret;

    ret = opal_common_ucx_atomic_fetch(ep, UCP_ATOMIC_FETCH_OP_CSWAP, compare, &tmp,
                                       op_size, remote_addr, rkey, worker);
    if (OPAL_LIKELY(OPAL_SUCCESS == ret)) {
        /* in case if op_size is constant (like sizeof(type)) then this condition
         * is evaluated in compile time */
        if (op_size == sizeof(uint64_t)) {
            *(uint64_t*)result = tmp;
        } else {
            assert(op_size == sizeof(uint32_t));
            *(uint32_t*)result = tmp;
        }
    }
    return ret;
}

END_C_DECLS

#endif
