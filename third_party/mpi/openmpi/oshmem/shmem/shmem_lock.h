/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 */
#ifndef SHMEM_LOCK_H
#define SHMEM_LOCK_H

#include "oshmem_config.h"

int shmem_lock_init(void);
int shmem_lock_finalize(void);

void _shmem_set_lock(void *lock, int lock_size);
int _shmem_test_lock(void *lock, int lock_size);
void _shmem_clear_lock(void *lock, int lock_size);


#endif /*SHMEM_LOCK_H*/
