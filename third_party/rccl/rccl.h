/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
 * Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_H_
#define NCCL_H_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define NCCL_MAJOR 2
#define NCCL_MINOR 18
#define NCCL_PATCH 3
#define NCCL_SUFFIX ""

#define NCCL_VERSION_CODE 21803
#define NCCL_VERSION(X,Y,Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

#define RCCL_BFLOAT16 1
#define RCCL_GATHER_SCATTER 1
#define RCCL_ALLTOALLV 1

#ifdef __cplusplus
extern "C" {
#endif

#include <limits.h>

/*! @brief      Opaque handle to communicator
    @details    A communicator contains information required to facilitate collective communications calls */
typedef struct ncclComm* ncclComm_t;
#define NCCL_COMM_NULL NULL

#define NCCL_UNIQUE_ID_BYTES 128
/*! @brief      Opaque unique id used to initialize communicators
    @details    The ncclUniqueId must be passed to all participating ranks */
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; /*!< Opaque array>*/} ncclUniqueId;

/*! @defgroup   rccl_result_code Result Codes
    @details    The various result codes that RCCL API calls may return
    @{ */

/*! @brief      Result type
    @details    Return codes aside from ncclSuccess indicate that a call has failed */
  typedef enum {
    ncclSuccess                 =  0, /*!< No error */
    ncclUnhandledCudaError      =  1, /*!< Unhandled HIP error */
    ncclSystemError             =  2, /*!< Unhandled system error */
    ncclInternalError           =  3, /*!< Internal Error - Please report to RCCL developers */
    ncclInvalidArgument         =  4, /*!< Invalid argument */
    ncclInvalidUsage            =  5, /*!< Invalid usage */
    ncclRemoteError             =  6, /*!< Remote process exited or there was a network error */
    ncclInProgress              =  7, /*!< RCCL operation in progress */
    ncclNumResults              =  8  /*!< Number of result types */
  } ncclResult_t;
/*! @} */

#define NCCL_CONFIG_UNDEF_INT INT_MIN
#define NCCL_CONFIG_UNDEF_PTR NULL
#define NCCL_SPLIT_NOCOLOR -1

/*! @defgroup   rccl_config_type Communicator Configuration
    @details    Structure that allows for customizing Communicator behavior via ncclCommInitRankConfig
    @{ */

/*! @brief      Communicator configuration
    @details    Users can assign value to attributes to specify the behavior of a communicator */
typedef struct ncclConfig_v21700 {
  /* attributes that users should never touch. */
  size_t size;                 /*!< Should not be touched */
  unsigned int magic;          /*!< Should not be touched */
  unsigned int version;        /*!< Should not be touched */
  /* attributes that users are able to customize. */
  int blocking;                /*!< Whether or not calls should block or not */
  int cgaClusterSize;          /*!< Cooperative group array cluster size */
  int minCTAs;                 /*!< Minimum number of cooperative thread arrays (blocks) */
  int maxCTAs;                 /*!< Maximum number of cooperative thread arrays (blocks) */
  const char *netName;         /*!< Force NCCL to use a specfic network */
  int splitShare;              /*!< Allow communicators to share resources */
} ncclConfig_t;

/* Config initializer must be assigned to initialize config structure when it is created.
 * Not initialized config will result in an error. */
#define NCCL_CONFIG_INITIALIZER {                                        \
  sizeof(ncclConfig_t),                             /* size */           \
  0xcafebeef,                                       /* magic */          \
  NCCL_VERSION(NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH), /* version */        \
  NCCL_CONFIG_UNDEF_INT,                            /* blocking */       \
  NCCL_CONFIG_UNDEF_INT,                            /* cgaClusterSize */ \
  NCCL_CONFIG_UNDEF_INT,                            /* minCTAs */        \
  NCCL_CONFIG_UNDEF_INT,                            /* maxCTAs */        \
  NCCL_CONFIG_UNDEF_PTR,                            /* netName */        \
  NCCL_CONFIG_UNDEF_INT                             /* splitShare */     \
}
/*! @} */

/*! @defgroup   rccl_api_version Version Information
    @details    API call that returns RCCL version
    @{ */

/*! @brief      Return the RCCL_VERSION_CODE of RCCL in the supplied integer.
    @details    This integer is coded with the MAJOR, MINOR and PATCH level of RCCL.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] version       Pointer to where version will be stored */
ncclResult_t  ncclGetVersion(int *version);
/*! @cond       include_hidden */
ncclResult_t pncclGetVersion(int *version);
/*! @endcond */
/*! @} */

/*! @defgroup   rccl_api_communicator Communicator Initialization/Destruction
    @details    API calls that operate on communicators.
                Communicators objects are used to launch collective communication
                operations.  Unique ranks between 0 and N-1 must be assigned to
                each HIP device participating in the same Communicator.
                Using the same HIP device for multiple ranks of the same Communicator
                is not supported at this time.
    @{ */

/*! @brief      Generates an ID for ncclCommInitRank.
    @details    Generates an ID to be used in ncclCommInitRank.
                ncclGetUniqueId should be called once by a single rank and the
                ID should be distributed to all ranks in the communicator before
                using it as a parameter for ncclCommInitRank.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] uniqueId      Pointer to where uniqueId will be stored */
ncclResult_t  ncclGetUniqueId(ncclUniqueId* uniqueId);
/*! @cond       include_hidden */
ncclResult_t pncclGetUniqueId(ncclUniqueId* uniqueId);
/*! @endcond */

/*! @brief      Create a new communicator with config.
    @details    Create a new communicator (multi thread/process version) with a configuration
                set by users. See @ref rccl_config_type for more details.
                Each rank is associated to a CUDA device, which has to be set before calling
                ncclCommInitRank.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] comm          Pointer to created communicator
    @param[in]  nranks        Total number of ranks participating in this communicator
    @param[in]  commId        UniqueId required for initialization
    @param[in]  rank          Current rank to create communicator for. [0 to nranks-1]
    @param[in]  config        Pointer to communicator configuration */
ncclResult_t  ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
/*! @cond       include_hidden */
ncclResult_t pncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
/*! @endcond */

/*! @brief      Creates a new communicator (multi thread/process version).
    @details    Rank must be between 0 and nranks-1 and unique within a communicator clique.
                Each rank is associated to a CUDA device, which has to be set before calling
                ncclCommInitRank.  ncclCommInitRank implicitly syncronizes with other ranks,
                so it must be called by different threads/processes or use ncclGroupStart/ncclGroupEnd.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] comm          Pointer to created communicator
    @param[in]  nranks        Total number of ranks participating in this communicator
    @param[in]  commId        UniqueId required for initialization
    @param[in]  rank          Current rank to create communicator for */
ncclResult_t  ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
/*! @cond       include_hidden */
ncclResult_t pncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
/*! @endcond */

/*! @brief      Creates a clique of communicators (single process version).
    @details    This is a convenience function to create a single-process communicator clique.
                Returns an array of ndev newly initialized communicators in comm.
                comm should be pre-allocated with size at least ndev*sizeof(ncclComm_t).
                If devlist is NULL, the first ndev HIP devices are used.
                Order of devlist defines user-order of processors within the communicator.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] comm          Pointer to array of created communicators
    @param[in]  ndev          Total number of ranks participating in this communicator
    @param[in]  devlist       Array of GPU device indices to create for */
ncclResult_t  ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
/*! @cond       include_hidden */
ncclResult_t pncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
/*! @endcond */

/*! @brief      Finalize a communicator.
    @details    ncclCommFinalize flushes all issued communications
                and marks communicator state as ncclInProgress. The state will change to ncclSuccess
                when the communicator is globally quiescent and related resources are freed; then,
                calling ncclCommDestroy can locally free the rest of the resources (e.g. communicator
                itself) without blocking.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to finalize */
ncclResult_t  ncclCommFinalize(ncclComm_t comm);
/*! @cond       include_hidden */
ncclResult_t pncclCommFinalize(ncclComm_t comm);
/*! @endcond */

/*! @brief      Frees local resources associated with communicator object.
    @details    Destroy all local resources associated with the passed in communicator object
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to destroy */
ncclResult_t  ncclCommDestroy(ncclComm_t comm);
/*! @cond       include_hidden */
ncclResult_t pncclCommDestroy(ncclComm_t comm);
/*! @endcond */

/*! @brief      Abort any in-progress calls and destroy the communicator object.
    @details    Frees resources associated with communicator object and aborts any operations
                that might still be running on the device.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to abort and destroy */
ncclResult_t  ncclCommAbort(ncclComm_t comm);
/*! @cond       include_hidden */
ncclResult_t pncclCommAbort(ncclComm_t comm);
/*! @endcond */

/*! @brief      Create one or more communicators from an existing one.
    @details    Creates one or more communicators from an existing one.
                Ranks with the same color will end up in the same communicator.
                Within the new communicator, key will be used to order ranks.
                NCCL_SPLIT_NOCOLOR as color will indicate the rank will not be part of any group
                and will therefore return a NULL communicator.
                If config is NULL, the new communicator will inherit the original communicator's configuration
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Original communicator object for this rank
    @param[in]  color         Color to assign this rank
    @param[in]  key           Key used to order ranks within the same new communicator
    @param[out] newcomm       Pointer to new communicator
    @param[in]  config        Config file for new communicator. May be NULL to inherit from comm */
ncclResult_t  ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);
/*! @cond       include_hidden */
ncclResult_t pncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t* config);
/*! @endcond */
/*! @} */

/*! @defgroup   rccl_api_errcheck Error Checking Calls
    @details    API calls that check for errors
    @{ */

/*! @brief      Returns a string for each result code.
    @details    Returns a human-readable string describing the given result code.
    @return     String containing description of result code.

    @param[in]  result        Result code to get description for */
const char*  ncclGetErrorString(ncclResult_t result);
/*! @cond       include_hidden */
const char* pncclGetErrorString(ncclResult_t result);
/*! @endcond */

/*! @brief      Returns mesage on last result that occured.
    @details    Returns a human-readable message of the last error that occurred.
    @return     String containing the last result

    @param[in]  comm is currently unused and can be set to NULL */
const char*  ncclGetLastError(ncclComm_t comm);
/*! @cond       include_hidden */
const char* pncclGetLastError(ncclComm_t comm);
/*! @endcond */

/*! @brief      Checks whether the comm has encountered any asynchronous errors
    @details    Query whether the provided communicator has encountered any asynchronous errors
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] asyncError    Pointer to where result code will be stored */
ncclResult_t  ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
/*! @cond       include_hidden */
ncclResult_t pncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError);
/*! @endcond */
/*! @} */

/*! @defgroup   rccl_api_comminfo Communicator Information
    @details    API calls that query communicator information
    @{ */

/*! @brief      Gets the number of ranks in the communicator clique.
    @details    Returns the number of ranks in the communicator clique (as set during initialization)
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] count         Pointer to where number of ranks will be stored */
ncclResult_t  ncclCommCount(const ncclComm_t comm, int* count);
/*! @cond       include_hidden */
ncclResult_t pncclCommCount(const ncclComm_t comm, int* count);
/*~ @endcond */

/*! @brief      Get the ROCm device index associated with a communicator
    @details    Returns the ROCm device number associated with the provided communicator.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] device        Pointer to where the associated ROCm device index will be stored */
ncclResult_t  ncclCommCuDevice(const ncclComm_t comm, int* device);
/*! @cond       include_hidden */
ncclResult_t pncclCommCuDevice(const ncclComm_t comm, int* device);
/*! @endcond */

/*! @brief      Get the rank associated with a communicator
    @details    Returns the user-ordered "rank" associated with the provided communicator.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  comm          Communicator to query
    @param[out] rank          Pointer to where the associated rank will be stored */
ncclResult_t  ncclCommUserRank(const ncclComm_t comm, int* rank);
/*! @cond       include_hidden */
ncclResult_t pncclCommUserRank(const ncclComm_t comm, int* rank);
/*! @endcond */
/*! @} */

/*! @defgroup   rccl_api_enumerations API Enumerations
    @details    Enumerations used by collective communication calls
    @{ */

/*! @brief      Dummy reduction enumeration
    @details    Dummy reduction enumeration used to determine value for ncclMaxRedOp */
typedef enum { ncclNumOps_dummy = 5 } ncclRedOp_dummy_t;

/*! @brief      Reduction operation selector
    @details    Enumeration used to specify the various reduction operations
                ncclNumOps is the number of built-in ncclRedOp_t values and serves as
                the least possible value for dynamic ncclRedOp_t values constructed by
                ncclRedOpCreate functions.

                ncclMaxRedOp is the largest valid value for ncclRedOp_t and is defined
                to be the largest signed value (since compilers are permitted to use
                signed enums) that won't grow sizeof(ncclRedOp_t) when compared to previous
                RCCL versions to maintain ABI compatibility. */
typedef enum { ncclSum        = 0, /*!< Sum */
               ncclProd       = 1, /*!< Product */
               ncclMax        = 2, /*!< Max */
               ncclMin        = 3, /*!< Min */
               ncclAvg        = 4, /*!< Average */
               ncclNumOps     = 5, /*!< Number of built-in reduction ops */
               ncclMaxRedOp   = 0x7fffffff>>(32-8*sizeof(ncclRedOp_dummy_t)) /*!< Largest value for ncclRedOp_t */
             } ncclRedOp_t;

/*! @brief      Data types
    @details    Enumeration of the various supported datatype */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclBfloat16   = 9,
               ncclNumTypes   = 10 } ncclDataType_t;
/*! @} */

/*! @defgroup   rccl_api_custom_redop Custom Reduction Operator
    @details    API calls relating to creation/destroying custom reduction operator
                that pre-multiplies local source arrays prior to reduction
    @{ */

/*! @brief      Location and dereferencing logic for scalar arguments.
    @details    Enumeration specifying memory location of the scalar argument.
                Based on where the value is stored, the argument will be dereferenced either
                while the collective is running (if in device memory), or before the ncclRedOpCreate()
                function returns (if in host memory). */
typedef enum {
  ncclScalarDevice        = 0, /*!< Scalar is in device-visible memory */
  ncclScalarHostImmediate = 1  /*!< Scalar is in host-visible memory */
} ncclScalarResidence_t;

/*! @brief      Create a custom pre-multiplier reduction operator
    @details    Creates a new reduction operator which pre-multiplies input values by a given
                scalar locally before reducing them with peer values via summation. For use
                only with collectives launched against *comm* and *datatype*. The
                *residence* argument indicates how/when the memory pointed to by *scalar*
                will be dereferenced. Upon return, the newly created operator's handle
                is stored in *op*.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] op            Pointer to where newly created custom reduction operator is to be stored
    @param[in]  scalar        Pointer to scalar value.
    @param[in]  datatype      Scalar value datatype
    @param[in]  residence     Memory type of the scalar value
    @param[in]  comm          Communicator to associate with this custom reduction operator */
ncclResult_t  ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
/*! @cond       include_hidden */
ncclResult_t pncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype, ncclScalarResidence_t residence, ncclComm_t comm);
/*! @endcond */

/*! @brief      Destroy custom reduction operator
    @details    Destroys the reduction operator *op*. The operator must have been created by
                ncclRedOpCreatePreMul with the matching communicator *comm*. An operator may be
                destroyed as soon as the last RCCL function which is given that operator returns.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  op            Custom reduction operator is to be destroyed
    @param[in]  comm          Communicator associated with this reduction operator */
ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
/*! @cond       include_hidden */
ncclResult_t pncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm);
/*! @endcond */
/*! @} */

/*! @defgroup   rccl_collective_api Collective Communication Operations
    @details    Collective communication operations must be called separately for each
                communicator in a communicator clique.

                They return when operations have been enqueued on the HIP stream.
                Since they may perform inter-CPU synchronization, each call has to be done
                from a different thread or process, or need to use Group Semantics (see
                below).
    @{ */

/*! @brief      Reduce
    @details    Reduces data arrays of length *count* in *sendbuff* into *recvbuff* using *op*
                operation.
                *recvbuff* may be NULL on all calls except for root device.
                *root* is the rank (not the HIP device) where data will reside after the
                 operation is complete.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Local device data buffer to be reduced
    @param[out] recvbuff      Data buffer where result is stored (only for *root* rank).  May be null for other ranks.
    @param[in]  count         Number of elements in every send buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator type
    @param[in]  root          Rank where result data array will be stored
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, int root, ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      (Deprecated) Broadcast (in-place)
    @details    Copies *count* values from *root* to all other devices.
                root is the rank (not the CUDA device) where data resides before the
                operation is started.
                This operation is implicitly in-place.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in,out]  buff      Input array on *root* to be copied to other ranks.  Output array for all ranks.
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank owning buffer to be copied to others
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      Broadcast
    @details    Copies *count* values from *sendbuff* on *root* to *recvbuff* on all devices.
                *root* is the rank (not the HIP device) where data resides before the operation is started.
                *sendbuff* may be NULL on ranks other than *root*.
                In-place operation will happen if *sendbuff* == *recvbuff*.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to copy (if *root*).  May be NULL for other ranks
    @param[in]  recvbuff      Data array to store received array
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank of broadcast root
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      All-Reduce
    @details    Reduces data arrays of length *count* in *sendbuff* using *op* operation, and
                leaves identical copies of result on each *recvbuff*.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Input data array to reduce
    @param[out] recvbuff      Data array to store reduced result array
    @param[in]  count         Number of elements in data buffer
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      Reduce-Scatter
    @details    Reduces data in *sendbuff* using *op* operation and leaves reduced result
                scattered over the devices so that *recvbuff* on rank i will contain the i-th
                block of the result.
                Assumes sendcount is equal to nranks*recvcount, which means that *sendbuff*
                should have a size of at least nranks*recvcount elements.
                In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Input data array to reduce
    @param[out] recvbuff      Data array to store reduced result subarray
    @param[in]  recvcount     Number of elements each rank receives
    @param[in]  datatype      Data buffer element datatype
    @param[in]  op            Reduction operator
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclReduceScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
    hipStream_t stream);
/*! @endcond */

/*! @brief      All-Gather
    @details    Each device gathers *sendcount* values from other GPUs into *recvbuff*,
                receiving data from rank i at offset i*sendcount.
                Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
                should have a size of at least nranks*sendcount elements.
                In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Input data array to send
    @param[out] recvbuff      Data array to store the gathered result
    @param[in]  sendcount     Number of elements each rank sends
    @param[in]  datatype      Data buffer element datatype
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      Send
    @details    Send data from *sendbuff* to rank *peer*.
                Rank *peer* needs to call ncclRecv with the same *datatype* and the same *count*
                as this rank.
                This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
                need to progress concurrently to complete, they must be fused within a ncclGroupStart /
                ncclGroupEnd section.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to send
    @param[in]  count         Number of elements to send
    @param[in]  datatype      Data buffer element datatype
    @param[in]  peer          Peer rank to send to
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      Receive
    @details    Receive data from rank *peer* into *recvbuff*.
                Rank *peer* needs to call ncclSend with the same datatype and the same count
                as this rank.
                This operation is blocking for the GPU. If multiple ncclSend and ncclRecv operations
                need to progress concurrently to complete, they must be fused within a ncclGroupStart/
                ncclGroupEnd section.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[out] recvbuff      Data array to receive
    @param[in]  count         Number of elements to receive
    @param[in]  datatype      Data buffer element datatype
    @param[in]  peer          Peer rank to send to
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      Gather
    @details    Root device gathers *sendcount* values from other GPUs into *recvbuff*,
                receiving data from rank i at offset i*sendcount.
                Assumes recvcount is equal to nranks*sendcount, which means that *recvbuff*
                should have a size of at least nranks*sendcount elements.
                In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
                *recvbuff* may be NULL on ranks other than *root*.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to send
    @param[out] recvbuff      Data array to receive into on *root*.
    @param[in]  sendcount     Number of elements to send per rank
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank that receives data from all other ranks
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, int root, ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      Scatter
    @details    Scattered over the devices so that recvbuff on rank i will contain the i-th
                block of the data on root.
                Assumes sendcount is equal to nranks*recvcount, which means that *sendbuff*
                should have a size of at least nranks*recvcount elements.
                In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to send (on *root* rank).  May be NULL on other ranks.
    @param[out] recvbuff      Data array to receive partial subarray into
    @param[in]  recvcount     Number of elements to receive per rank
    @param[in]  datatype      Data buffer element datatype
    @param[in]  root          Rank that scatters data to all other ranks
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, int root, ncclComm_t comm,
    hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclScatter(const void* sendbuff, void* recvbuff,
    size_t recvcount, ncclDataType_t datatype, int root, ncclComm_t comm,
    hipStream_t stream);
/*! @endcond */

/*! @brief      All-To-All
    @details    Device (i) send (j)th block of data to device (j) and be placed as (i)th
                block. Each block for sending/receiving has *count* elements, which means
                that *recvbuff* and *sendbuff* should have a size of nranks*count elements.
                In-place operation is NOT supported. It is the user's responsibility
                to ensure that sendbuff and recvbuff are distinct.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to send (contains blocks for each other rank)
    @param[out] recvbuff      Data array to receive (contains blocks from each other rank)
    @param[in]  count         Number of elements to send between each pair of ranks
    @param[in]  datatype      Data buffer element datatype
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclAllToAll(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      All-To-Allv
    @details    Device (i) sends sendcounts[j] of data from offset sdispls[j]
                to device (j). At the same time, device (i) receives recvcounts[j] of data
                from device (j) to be placed at rdispls[j].
                sendcounts, sdispls, recvcounts and rdispls are all measured in the units
                of datatype, not bytes.
                In-place operation will happen if sendbuff == recvbuff.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendbuff      Data array to send (contains blocks for each other rank)
    @param[in]  sendcounts    Array containing number of elements to send to each participating rank
    @param[in]  sdispls       Array of offsets into *sendbuff* for each participating rank
    @param[out] recvbuff      Data array to receive (contains blocks from each other rank)
    @param[in]  recvcounts    Array containing number of elements to receive from each participating rank
    @param[in]  rdispls       Array of offsets into *recvbuff* for each participating rank
    @param[in]  datatype      Data buffer element datatype
    @param[in]  comm          Communicator group object to execute on
    @param[in]  stream        HIP stream to execute collective on */
ncclResult_t  ncclAllToAllv(const void *sendbuff, const size_t sendcounts[],
    const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
    const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pncclAllToAllv(const void *sendbuff, const size_t sendcounts[],
    const size_t sdispls[], void *recvbuff, const size_t recvcounts[],
    const size_t rdispls[], ncclDataType_t datatype, ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @} */

/*! @defgroup   msccl_api MSCCL Algorithm
    @details    API calls relating to the optional MSCCL algorithm datapath
    @{ */

/*! @brief      Opaque handle to MSCCL algorithm */
typedef int mscclAlgoHandle_t;

/*! @brief      MSCCL Load Algorithm
    @details    Load MSCCL algorithm file specified in mscclAlgoFilePath and return
                its handle via mscclAlgoHandle. This API is expected to be called by MSCCL
                scheduler instead of end users.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  mscclAlgoFilePath  Path to MSCCL algorithm file
    @param[out] mscclAlgoHandle    Returned handle to MSCCL algorithm
    @param[in]  rank               Current rank */
ncclResult_t  mscclLoadAlgo(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank);
/*! @cond       include_hidden */
ncclResult_t pmscclLoadAlgo(const char *mscclAlgoFilePath, mscclAlgoHandle_t *mscclAlgoHandle, int rank);
/*! @endcond */

/*! @brief      MSCCL Run Algorithm
    @details    Run MSCCL algorithm specified by mscclAlgoHandle. The parameter
                list merges all possible parameters required by different operations as this
                is a general-purposed API. This API is expected to be called by MSCCL
                scheduler instead of end users.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  sendBuff         Data array to send
    @param[in]  sendCounts       Array containing number of elements to send to each participating rank
    @param[in]  sDisPls          Array of offsets into *sendbuff* for each participating rank
    @param[out] recvBuff         Data array to receive
    @param[in]  recvCounts       Array containing number of elements to receive from each participating rank
    @param[in]  rDisPls          Array of offsets into *recvbuff* for each participating rank
    @param[in]  count            Number of elements
    @param[in]  dataType         Data buffer element datatype
    @param[in]  root             Root rank index
    @param[in]  peer             Peer rank index
    @param[in]  op               Reduction operator
    @param[in]  mscclAlgoHandle  Handle to MSCCL algorithm
    @param[in]  comm             Communicator group object to execute on
    @param[in]  stream           HIP stream to execute collective on */
ncclResult_t  mscclRunAlgo(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream);
/*! @cond       include_hidden */
ncclResult_t pmscclRunAlgo(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclAlgoHandle_t mscclAlgoHandle, ncclComm_t comm, hipStream_t stream);
/*! @endcond */

/*! @brief      MSCCL Unload Algorithm
    @details    Unload MSCCL algorithm previous loaded using its handle. This API
                is expected to be called by MSCCL scheduler instead of end users.
    @return     Result code. See @ref rccl_result_code for more details.

    @param[in]  mscclAlgoHandle  Handle to MSCCL algorithm to unload
*/
ncclResult_t  mscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);
/*! @cond       include_hidden */
ncclResult_t pmscclUnloadAlgo(mscclAlgoHandle_t mscclAlgoHandle);
/*! @endcond */
/*! @} */


/*! @defgroup   rccl_group_api Group semantics
    @details    When managing multiple GPUs from a single thread, and since RCCL collective
                calls may perform inter-CPU synchronization, we need to "group" calls for
                different ranks/devices into a single call.

                Grouping RCCL calls as being part of the same collective operation is done
                using ncclGroupStart and ncclGroupEnd. ncclGroupStart will enqueue all
                collective calls until the ncclGroupEnd call, which will wait for all calls
                to be complete. Note that for collective communication, ncclGroupEnd only
                guarantees that the operations are enqueued on the streams, not that
                the operation is effectively done.

                Both collective communication and ncclCommInitRank can be used in conjunction
                of ncclGroupStart/ncclGroupEnd, but not together.

                Group semantics also allow to fuse multiple operations on the same device
                to improve performance (for aggregated collective calls), or to permit
                concurrent progress of multiple send/receive operations.
    @{ */

/*! @brief      Group Start
    @details    Start a group call. All calls to RCCL until ncclGroupEnd will be fused into
                a single RCCL operation. Nothing will be started on the HIP stream until
                ncclGroupEnd.
    @return     Result code. See @ref rccl_result_code for more details. */
ncclResult_t  ncclGroupStart();
/*! @cond       include_hidden */
ncclResult_t pncclGroupStart();
/*! @endcond */

/*! @brief      Group End
    @details    End a group call. Start a fused RCCL operation consisting of all calls since
                ncclGroupStart. Operations on the HIP stream depending on the RCCL operations
                need to be called after ncclGroupEnd.
    @return     Result code. See @ref rccl_result_code for more details. */
ncclResult_t  ncclGroupEnd();
/*! @cond       include_hidden */
ncclResult_t pncclGroupEnd();
/*! @endcond */
/*! @} */

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // end include guard
