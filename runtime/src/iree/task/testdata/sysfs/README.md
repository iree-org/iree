# Sysfs CPU Topology Test Data

This directory contains snapshots of `/sys/devices/system/cpu/` directory
structures from various systems. These snapshots are used to test the
`topology_sysfs.c` implementation without requiring physical hardware access.

**Note:** Test data is stored as compressed tar.gz archives to minimize
repository size. Extract archives before testing:

```bash
tar xzf arm64_pixel6_tensor.tar.gz
```

We only check in small weird configurations (like ARM) for smoke testing and
manual debugging. Large x86 systems can be thousands of files and megabytes of
text - capture those locally for testing but don't check them in.

## Test Configurations

### arm64_pixel6_tensor/

**Hardware:** Google Pixel 6 (Google Tensor GS101)
**Architecture:** ARM64
**Configuration:**
- 8 cores in heterogeneous big.LITTLE configuration:
  - CPUs 0-1: Cortex-X1 (prime), capacity 1024, cluster 0
  - CPUs 2-3: Cortex-A76 (big), capacity 820, cluster 1
  - CPUs 4-7: Cortex-A55 (LITTLE), capacity 280, cluster 2
- Cache hierarchy varies by cluster:
  - X1: L1 32KB, L2 512KB, L3 4MB (shared 0-1)
  - A76: L1 32KB, L2 256KB, L3 4MB (shared 2-3)
  - A55: L1 32KB, L2 128KB, L3 4MB (shared 4-7)

**Expected Behavior:**
- Should detect 3 clusters based on cluster_id
- With 75% capacity threshold (768):
  - HIGH performance: CPUs 0-1, 2-3 (capacity >= 768)
  - LOW performance: CPUs 4-7 (capacity < 768)
  - ANY: All 8 CPUs
- Heterogeneous system should be detected (max_capacity 1024 != min_capacity 280)

## Capturing New Test Data

Use the `capture_sysfs.sh` script to create snapshots from real systems:

```bash
# Capture current system to a new directory
./capture_sysfs.sh my_system_name

# Capture with automatic timestamped name
./capture_sysfs.sh
```

The script captures:
- Top-level CPU list files (`present`, `possible`, `online`, etc.)
- Per-CPU topology (`core_id`, `cluster_id`, `physical_package_id`, etc.)
- Per-CPU cache hierarchy (`type`, `level`, `size`, `shared_cpu_list`)
- ARM-specific files (`cpu_capacity` for big.LITTLE detection)

## Testing with Snapshots

To test with sysfs snapshots, compile IREE with the `IREE_SYSFS_ROOT` define
pointing to your snapshot directory:

```bash
# Configure with sysfs snapshot path
cmake -B build/ -S . \
  -DCMAKE_C_FLAGS="-DIREE_SYSFS_ROOT=\\\"/path/to/arm64_pixel6_tensor\\\"" \
  -DCMAKE_CXX_FLAGS="-DIREE_SYSFS_ROOT=\\\"/path/to/arm64_pixel6_tensor\\\""

# Build and test
cmake --build build/ --target iree-run-module

# Test ARM64 big.LITTLE with default settings (all cores, scatter distribution)
./build/tools/iree-run-module --dump_task_topologies

# Test ARM64 big.LITTLE with HIGH performance cores only
./build/tools/iree-run-module \
  --task_topology_performance_level=high \
  --dump_task_topologies

# Test ARM64 with compact distribution (fill cache domains sequentially)
./build/tools/iree-run-module \
  --task_topology_distribution=compact \
  --dump_task_topologies

# Test ARM64 with latency preset (compact + high performance)
./build/tools/iree-run-module \
  --task_topology_favor=latency \
  --dump_task_topologies

# Test ARM64 with only LITTLE cores (low power)
./build/tools/iree-run-module \
  --task_topology_performance_level=low \
  --dump_task_topologies
```

Available flags for topology configuration:
- `--task_topology_performance_level`: `any`, `low` (or `efficiency`), `high` (or `performance`)
- `--task_topology_distribution`: `compact`, `scatter`
- `--task_topology_favor`: `latency`, `throughput`, `efficiency` (overrides above flags)
- `--task_topology_nodes`: `current`, `all`, or comma-separated node IDs (e.g., `0,2`)

## Validation

When testing, verify:

1. **CPU detection:** Correct number of logical processors detected
2. **Cluster/node mapping:** CPUs grouped correctly by cluster_id
3. **Cache hierarchy:** L1/L2/L3 sizes and sharing masks accurate
4. **ARM big.LITTLE filtering:**
   - ANY mode includes all cores
   - HIGH mode filters to high-capacity cores only
   - LOW mode filters to low-capacity cores only
5. **Graceful degradation:** Missing files handled without errors

## Adding New Test Cases

Good candidates for test data to capture locally (but only check in small/unusual ones):

- **NUMA systems:** Multi-socket x86 servers (Xeon, EPYC) - capture locally, don't check in
- **ARM heterogeneous:** Snapdragon 8 Gen 3, Apple M-series, AWS Graviton - small enough to check in
- **Unusual configurations:** Hybrid x86 (Intel Alder Lake P/E cores), RISC-V SMP - check in if interesting

When adding new test data:
1. Run `capture_sysfs.sh <directory_name>` on the target hardware
2. Compress the directory: `tar czf <directory_name>.tar.gz <directory_name>/`
3. Test the snapshot to ensure it exercises the intended code paths
4. **Only check in tar.gz if small and interesting** (ARM heterogeneous, exotic architectures)
5. Update this README with hardware details and expected behavior if checking in
6. Large x86 systems stay local - don't commit multi-MB tar files
