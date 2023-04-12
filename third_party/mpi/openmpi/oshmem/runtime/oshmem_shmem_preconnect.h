/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_SHMEM_PRECONNECT_H
#define OSHMEM_SHMEM_PRECONNECT_H

BEGIN_C_DECLS

/** Preconnect peers */
int oshmem_shmem_preconnect_all(void);

/** Finalize preconnection framework*/
int oshmem_shmem_preconnect_all_finalize(void);

END_C_DECLS

#endif
