/* oshmem/include/shmem-compat.h. This file contains OpenSHMEM lagacy API  */
/*
 * Copyright (c) 2014-2017 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_SHMEM_COMPAT_H
#define OSHMEM_SHMEM_COMPAT_H

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

/*
 * Legacy API
 * old init/destruct functions - not in the open shmem spec but still supported
 */
OSHMEM_DECLSPEC  int num_pes(void);
OSHMEM_DECLSPEC  int my_pe(void);

OSHMEM_DECLSPEC  void start_pes(int npes);

OSHMEM_DECLSPEC  int _num_pes(void);
OSHMEM_DECLSPEC  int _my_pe(void);

OSHMEM_DECLSPEC  void* shmalloc(size_t size);
OSHMEM_DECLSPEC  void* shmemalign(size_t align, size_t size);
OSHMEM_DECLSPEC  void* shrealloc(void *ptr, size_t size);
OSHMEM_DECLSPEC  void shfree(void* ptr);

OSHMEM_DECLSPEC  void globalexit(int status);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif /* OSHMEM_SHMEM_COMPAT_H */
