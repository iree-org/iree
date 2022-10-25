#include "iree/task/numactl.h"

char *fmt_mem(unsigned long long mem, char *buf) {
	if (mem == -1L)
		sprintf(buf, "<not available>");
	else
		sprintf(buf, "%llu MB", mem >> 20);
	return buf;
}

void print_distances(int maxnode) {
	int i,k;
	int fst = 0;

	for (i = 0; i <= maxnode; i++) {
		if (numa_bitmask_isbitset(numa_nodes_ptr, i)) {
			fst = i;
			break;
		}
	}
	if (numa_distance(maxnode,fst) == 0) {
		printf("No distance information available.\n");
		return;
	}
	printf("node distances:\n");
	printf("node ");
	for (i = 0; i <= maxnode; i++)
		if (numa_bitmask_isbitset(numa_nodes_ptr, i))
			printf("% 3d ", i);
	printf("\n");
	for (i = 0; i <= maxnode; i++) {
		if (!numa_bitmask_isbitset(numa_nodes_ptr, i))
			continue;
		printf("% 3d: ", i);
		for (k = 0; k <= maxnode; k++)
			if (numa_bitmask_isbitset(numa_nodes_ptr, i) &&
			    numa_bitmask_isbitset(numa_nodes_ptr, k))
				printf("% 3d ", numa_distance(i,k));
		printf("\n");
	}
}

void print_node_cpus(int node) {
	int i, err;
	struct bitmask *cpus;

	cpus = numa_allocate_cpumask();
	err = numa_node_to_cpus(node, cpus);
	if (err >= 0) {
		for (i = 0; i < cpus->size; i++)
			if (numa_bitmask_isbitset(cpus, i))
				printf(" %d", i);
	}
	putchar('\n');
}

void hardware_test(void) {
	int i;
	int numnodes=0;
	int prevnode=-1;
	int skip=0;
	int maxnode = numa_max_node();

	if (numa_available() < 0) {
                printf("No NUMA available on this system\n");
                exit(1);
        }

	for (i=0; i<=maxnode; i++)
		if (numa_bitmask_isbitset(numa_nodes_ptr, i))
			numnodes++;
	printf("available: %d nodes (", numnodes);
	for (i=0; i<=maxnode; i++) {
		if (numa_bitmask_isbitset(numa_nodes_ptr, i)) {
			if (prevnode == -1) {
				printf("%d", i);
				prevnode=i;
				continue;
			}

			if (i > prevnode + 1) {
				if (skip) {
					printf("%d", prevnode);
					skip=0;
				}
				printf(",%d", i);
				prevnode=i;
				continue;
			}

			if (i == prevnode + 1) {
				if (!skip) {
					printf("-");
					skip=1;
				}
				prevnode=i;
			}

			if ((i == maxnode) && skip)
				printf("%d", prevnode);
		}
	}
	printf(")\n");

	for (i = 0; i <= maxnode; i++) {
		char buf[64];
		long long fr;
		unsigned long long sz = numa_node_size64(i, &fr);
		if (!numa_bitmask_isbitset(numa_nodes_ptr, i))
			continue;

		printf("node %d cpus:", i);
		print_node_cpus(i);
		printf("node %d size: %s\n", i, fmt_mem(sz, buf));
		printf("node %d free: %s\n", i, fmt_mem(fr, buf));
	}
	print_distances(maxnode);
}