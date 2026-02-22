// Scenario definitions for the Vector Clock & Frontier Simulator.
//
// Each scenario describes a set of hardware lanes (physical execution
// resources), semaphores (the clock axes for frontier propagation), and
// operations (the work graph). Operations are scheduled onto hardware lanes and
// ordered by semaphore waits/signals. Hardware sharing is a scheduling
// constraint; semaphores are the ONLY causal ordering mechanism.
//
// Scenario format:
//   hardware:   [{id, label}]      — physical execution lanes
//   semaphores: [{id, label}]      — clock axes / frontier dimensions
//   operations: [{id, hardware, label, duration, wait:{}, signal:{}}]
//   annotations: [{tick, text}]    — educational commentary per tick

export const scenarios = [

  // -------------------------------------------------------------------------
  // 1. Sequential Queue — one device, one semaphore, linear chain
  // -------------------------------------------------------------------------
  {
    id: 'sequential',
    name: '1. Sequential Queue',
    chain_through: ['q'],
    description:
        'The simplest case: three operations run one after another on a single ' +
        'GPU, ordered by a single semaphore. Each operation waits for the previous ' +
        'one to signal before starting. The frontier is just the latest signal value — ' +
        'vector clocks become interesting when there are multiple semaphores.',
    hardware: [
      {id: 'gpu0', label: 'GPU'},
    ],
    semaphores: [
      {id: 'q', label: 'Queue'},
    ],
    operations: [
      {
        id: 'fill',
        hardware: 'gpu0',
        label: 'Fill',
        duration: 2,
        wait: {},
        signal: {q: 1}
      },
      {
        id: 'matmul',
        hardware: 'gpu0',
        label: 'MatMul',
        duration: 3,
        wait: {q: 1},
        signal: {q: 2}
      },
      {
        id: 'readback',
        hardware: 'gpu0',
        label: 'Readback',
        duration: 1,
        wait: {q: 2},
        signal: {q: 3}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'Fill starts with no dependencies. Its frontier is {q:1} \u2014 the semaphore ' +
            'value it will signal, representing its own contribution to the timeline.'
      },
      {
        tick: 2,
        text:
            'Fill completes and signals q to 1. MatMul\'s wait on q \u2265 1 is satisfied. ' +
            'MatMul\'s frontier inherits Fill\'s knowledge through semaphore q and adds ' +
            'its own signal: {q:2}.'
      },
      {
        tick: 5,
        text:
            'MatMul completes. Readback starts. Each operation\'s frontier is the ' +
            'component-wise maximum of all waited semaphore frontiers plus its own ' +
            'signal values.'
      },
      {
        tick: 6,
        text:
            'All operations retired. With a single semaphore, the frontier is just ' +
            'the latest signal value. The real power of vector clocks emerges with ' +
            'multiple semaphores tracking independent causal streams.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 2. Cross-Queue Dependency — two devices, semaphore-mediated dependency
  // -------------------------------------------------------------------------
  {
    id: 'cross-queue',
    name: '2. Cross-Queue Dependency',
    description:
        'Operations across a GPU pool with a cross-stream dependency. MatMul ' +
        'produces a result that Copy needs. Both Activation and Copy wait on the ' +
        'same semaphore \u2014 the scheduler runs them in parallel when GPUs are ' +
        'available. Copy\'s frontier carries transitive knowledge of MatMul. ' +
        'Try the GPU slider: with 1 GPU everything serializes, but frontiers ' +
        'are identical.',
    hardware: [
      {id: 'gpu', label: 'GPU', count: 2},
    ],
    semaphores: [
      {id: 'A', label: 'Stream A'},
      {id: 'B', label: 'Stream B'},
    ],
    operations: [
      {
        id: 'matmul',
        hardware: 'gpu',
        label: 'MatMul',
        duration: 4,
        wait: {},
        signal: {A: 1}
      },
      {
        id: 'activation',
        hardware: 'gpu',
        label: 'Activation',
        duration: 2,
        wait: {A: 1},
        signal: {A: 2}
      },
      {
        id: 'copy',
        hardware: 'gpu',
        label: 'Copy',
        duration: 3,
        wait: {A: 1},
        signal: {B: 1}
      },
      {
        id: 'process',
        hardware: 'gpu',
        label: 'Process',
        duration: 2,
        wait: {B: 1},
        signal: {B: 2}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'MatMul starts on GPU 0 with no dependencies. GPU 1 is idle \u2014 no ' +
            'operations are ready yet.'
      },
      {
        tick: 4,
        text:
            'MatMul completes and signals A to 1. Both Activation and Copy were ' +
            'waiting on A \u2265 1 \u2014 they start simultaneously on their respective GPUs. ' +
            'Copy\'s frontier {A:1, B:1} carries knowledge of MatMul\'s work through ' +
            'semaphore A, even though Copy runs on a different device.'
      },
      {
        tick: 7,
        text:
            'Copy completes on GPU 1, signaling B to 1. Process starts and inherits ' +
            'Copy\'s frontier, which includes A:1 \u2014 transitive knowledge of MatMul, ' +
            'even though Process never directly waited on semaphore A.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 3. Fork-Join — fan-out and fan-in, frontier merge at join point
  // -------------------------------------------------------------------------
  {
    id: 'fork-join',
    name: '3. Fork-Join',
    chain_through: ['s'],
    description:
        'A common pattern: one operation fans out to two parallel branches, which ' +
        'later fan back in. The Merge operation waits on BOTH branches \u2014 its ' +
        'frontier is the component-wise maximum of both paths. This is where ' +
        'vector clocks shine: the merge captures the full causal history of ' +
        'both branches in a single frontier. Try 1 GPU: branches serialize, but ' +
        'the frontier is identical.',
    hardware: [
      {id: 'gpu', label: 'GPU', count: 2},
    ],
    semaphores: [
      {id: 's', label: 'Main'},
      {id: 'a', label: 'Branch A'},
      {id: 'b', label: 'Branch B'},
    ],
    operations: [
      {
        id: 'setup',
        hardware: 'gpu',
        label: 'Setup',
        duration: 2,
        wait: {},
        signal: {s: 1}
      },
      {
        id: 'branch_a',
        hardware: 'gpu',
        label: 'Branch A',
        duration: 3,
        wait: {s: 1},
        signal: {a: 1}
      },
      {
        id: 'branch_b',
        hardware: 'gpu',
        label: 'Branch B',
        duration: 4,
        wait: {s: 1},
        signal: {b: 1}
      },
      {
        id: 'merge',
        hardware: 'gpu',
        label: 'Merge',
        duration: 2,
        wait: {a: 1, b: 1},
        signal: {s: 2}
      },
    ],
    annotations: [
      {tick: 0, text: 'Setup begins the work graph on GPU 0.'},
      {
        tick: 2,
        text:
            'Setup completes, enabling both branches in parallel on different GPUs. ' +
            'Each branch inherits Setup\'s frontier through semaphore s. Branch A ' +
            'runs on GPU 0, Branch B on GPU 1.'
      },
      {
        tick: 5,
        text:
            'Branch A completes first, but Merge must wait for BOTH branches \u2014 the ' +
            'multi-wait is the join point. GPU 0 is idle, waiting for Branch B.'
      },
      {
        tick: 6,
        text:
            'Branch B completes. Merge starts. Its frontier is the component-wise ' +
            'maximum of both branches\' semaphore frontiers: {s:2, a:1, b:1}. This ' +
            'single vector captures the full causal history of both parallel paths.'
      },
      {
        tick: 8,
        text:
            'Merge retires. Anything downstream waiting on s \u2265 2 inherits knowledge ' +
            'of both branches through a single semaphore wait \u2014 no need to enumerate ' +
            'the individual branch semaphores.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 4. Independent Streams — same hardware, NO frontier leakage
  // -------------------------------------------------------------------------
  {
    id: 'independent',
    name: '4. Independent Streams',
    description:
        'Two completely independent workloads sharing GPU resources. Despite ' +
        'running on the same hardware, their frontiers have zero overlap. This ' +
        'demonstrates a critical property: hardware scheduling is NOT a causal ' +
        'relationship. Semaphores are the only thing that creates causal ordering. ' +
        'Try 2 GPUs: both workloads run in parallel, half the time, but the ' +
        'frontiers are identical \u2014 no causal leakage regardless of hardware count.',
    hardware: [
      {id: 'gpu', label: 'GPU', count: 1},
    ],
    semaphores: [
      {id: 'A', label: 'Workload A'},
      {id: 'B', label: 'Workload B'},
    ],
    operations: [
      {
        id: 'a1',
        hardware: 'gpu',
        label: 'A:Step 1',
        duration: 2,
        wait: {},
        signal: {A: 1}
      },
      {
        id: 'a2',
        hardware: 'gpu',
        label: 'A:Step 2',
        duration: 2,
        wait: {A: 1},
        signal: {A: 2}
      },
      {
        id: 'b1',
        hardware: 'gpu',
        label: 'B:Step 1',
        duration: 2,
        wait: {},
        signal: {B: 1}
      },
      {
        id: 'b2',
        hardware: 'gpu',
        label: 'B:Step 2',
        duration: 2,
        wait: {B: 1},
        signal: {B: 2}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'Two independent workloads compete for one GPU. A:Step 1 starts first ' +
            '(scheduling tiebreak by definition order). B:Step 1 is ready but the ' +
            'hardware lane is occupied. This is a scheduling constraint, NOT causal ordering.'
      },
      {
        tick: 2,
        text:
            'A:Step 1 completes. A:Step 2 gets the GPU next (same tiebreak). B:Step 1 ' +
            'continues waiting. The scheduling order is A1 \u2192 A2 \u2192 B1 \u2192 B2, but this ' +
            'creates NO causal relationship between A and B.'
      },
      {
        tick: 4,
        text:
            'A:Step 2 completes. B:Step 1 finally runs. Its frontier is {B:1} \u2014 there ' +
            'is NO trace of semaphore A. Despite executing after A:Step 2 on the same ' +
            'physical hardware, B has zero knowledge of A\'s work.'
      },
      {
        tick: 8,
        text:
            'Both workloads complete. A\'s frontier: {A:2}. B\'s frontier: {B:2}. ' +
            'Zero frontier leakage between independent workloads sharing hardware. ' +
            'With separate GPUs they would overlap in time, but the causal structure ' +
            'would be identical.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 5. Producer-Consumer Pipeline — pipelining with cross-stream waits
  // -------------------------------------------------------------------------
  {
    id: 'pipeline',
    name: '5. Producer-Consumer Pipeline',
    chain_through: ['C'],
    description:
        'A producer generates three chunks while a consumer processes them. ' +
        'Each consumer step waits on BOTH the next producer chunk (cross-stream) ' +
        'AND the previous consumer step (intra-stream ordering). Pipeline stages ' +
        'need explicit intra-stream waits \u2014 there is no implicit FIFO ordering.',
    hardware: [
      {id: 'prod', label: 'Producer'},
      {id: 'cons', label: 'Consumer'},
    ],
    semaphores: [
      {id: 'P', label: 'Producer'},
      {id: 'C', label: 'Consumer'},
    ],
    operations: [
      {
        id: 'p1',
        hardware: 'prod',
        label: 'P:Chunk 1',
        duration: 2,
        wait: {},
        signal: {P: 1}
      },
      {
        id: 'p2',
        hardware: 'prod',
        label: 'P:Chunk 2',
        duration: 2,
        wait: {P: 1},
        signal: {P: 2}
      },
      {
        id: 'p3',
        hardware: 'prod',
        label: 'P:Chunk 3',
        duration: 2,
        wait: {P: 2},
        signal: {P: 3}
      },
      {
        id: 'c1',
        hardware: 'cons',
        label: 'C:Chunk 1',
        duration: 3,
        wait: {P: 1},
        signal: {C: 1}
      },
      {
        id: 'c2',
        hardware: 'cons',
        label: 'C:Chunk 2',
        duration: 3,
        wait: {P: 2, C: 1},
        signal: {C: 2}
      },
      {
        id: 'c3',
        hardware: 'cons',
        label: 'C:Chunk 3',
        duration: 3,
        wait: {P: 3, C: 2},
        signal: {C: 3}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'P:Chunk 1 starts producing the first chunk. The consumer has nothing to consume yet.'
      },
      {
        tick: 2,
        text:
            'P:Chunk 1 done. P:Chunk 2 starts (waits on P \u2265 1 for intra-stream ordering). ' +
            'C:Chunk 1 starts consuming (also waits on P \u2265 1). Producer and consumer overlap ' +
            '\u2014 pipeline parallelism via semaphores.'
      },
      {
        tick: 4,
        text:
            'P:Chunk 2 done, P:Chunk 3 starts. The producer runs ahead of the consumer.'
      },
      {
        tick: 5,
        text:
            'C:Chunk 1 done. C:Chunk 2 waits on BOTH P \u2265 2 (cross-stream: needs chunk 2 ' +
            'data) AND C \u2265 1 (intra-stream: sequential consumer ordering). Both are ' +
            'satisfied \u2014 C:Chunk 2 starts immediately.'
      },
      {
        tick: 8,
        text:
            'C:Chunk 2 done. C:Chunk 3\'s frontier captures the full pipeline history: ' +
            '{P:3, C:3}. Every producer chunk and every consumer step are represented. ' +
            'Downstream work waiting on C \u2265 3 inherits the complete causal chain.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 6. Late Waiter — >= monotonic advance
  // -------------------------------------------------------------------------
  {
    id: 'late-waiter',
    name: '6. \u2265 Monotonic Advance',
    description:
        'Demonstrates how \u2265 semantics decouple scheduling from causality. A chain ' +
        'of operations advances semaphore s to value 3. Meanwhile, a slow setup ' +
        'runs on another GPU. When LateJoin becomes ready, its wait on s \u2265 1 ' +
        'is immediately satisfied (s is at 3). But LateJoin\'s frontier only ' +
        'includes the frontier from the s:1 signal \u2014 Step 2 and Step 3 are not ' +
        'in its causal past. Scheduling resolves instantly; causality is precise. ' +
        'Try 1 GPU: everything serializes, but the \u2265 property still holds.',
    hardware: [
      {id: 'gpu', label: 'GPU', count: 2},
    ],
    semaphores: [
      {id: 's', label: 'Pipeline'},
      {id: 'g', label: 'Gate'},
    ],
    operations: [
      {
        id: 'step1',
        hardware: 'gpu',
        label: 'Step 1',
        duration: 2,
        wait: {},
        signal: {s: 1}
      },
      {
        id: 'step2',
        hardware: 'gpu',
        label: 'Step 2',
        duration: 2,
        wait: {s: 1},
        signal: {s: 2}
      },
      {
        id: 'step3',
        hardware: 'gpu',
        label: 'Step 3',
        duration: 2,
        wait: {s: 2},
        signal: {s: 3}
      },
      {
        id: 'setup',
        hardware: 'gpu',
        label: 'Slow Setup',
        duration: 8,
        wait: {},
        signal: {g: 1}
      },
      {
        id: 'join',
        hardware: 'gpu',
        label: 'LateJoin',
        duration: 2,
        wait: {s: 1, g: 1},
        signal: {g: 2}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'Two independent hardware lanes start working. Step 1 begins a three-step ' +
            'chain on GPU 0 while Slow Setup runs a long preparation on GPU 1.'
      },
      {
        tick: 2,
        text:
            'Step 1 signals s to 1. Step 2 starts. The pipeline advances on GPU 0.'
      },
      {
        tick: 6,
        text:
            'Step 3 completes. Semaphore s is now at value 3. GPU 0 is idle \u2014 all ' +
            'its work is done. Slow Setup still running on GPU 1.'
      },
      {
        tick: 8,
        text:
            'Slow Setup finally completes. LateJoin waits on s \u2265 1: the semaphore ' +
            'is at 3, so the scheduling wait is immediately satisfied \u2014 no blocking. ' +
            'But LateJoin\'s frontier is {s:1, g:2}, NOT {s:3, g:2}. It inherits the ' +
            'frontier from the s:1 signal (Step 1) because that is its causal dependency. ' +
            'Steps 2 and 3 advanced s further, but LateJoin has no causal relationship ' +
            'with them. The \u2265 property decouples scheduling (the wait resolves instantly) ' +
            'from causality (the frontier tracks exactly what was depended on).'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 7. Multi-Domain DAG — transitive frontier propagation
  // -------------------------------------------------------------------------
  {
    id: 'multi-domain',
    name: '7. Multi-Domain DAG',
    chain_through: ['B'],
    description:
        'The crown jewel: two workloads (A and B) across CPU, two GPUs, and a ' +
        'NIC. B:Local depends on A:Compute\'s result, creating a cross-workload ' +
        'join. When B:Reduce merges local and remote results, its frontier ' +
        'transitively contains A\'s compute work (via B:Local \u2192 s_B \u2192 B:Reduce) ' +
        'even though B:Reduce never directly waited on A\'s semaphore.',
    hardware: [
      {id: 'cpu', label: 'CPU'},
      {id: 'gpu0', label: 'GPU 0 (local)'},
      {id: 'gpu1', label: 'GPU 1 (remote)'},
      {id: 'nic', label: 'NIC'},
    ],
    semaphores: [
      {id: 'A', label: 'Workload A'},
      {id: 'B', label: 'Workload B'},
      {id: 'R', label: 'Remote'},
    ],
    operations: [
      // Workload A: tokenize on CPU, compute on local GPU.
      {
        id: 'a_tok',
        hardware: 'cpu',
        label: 'A:Tokenize',
        duration: 2,
        wait: {},
        signal: {A: 1}
      },
      {
        id: 'a_compute',
        hardware: 'gpu0',
        label: 'A:Compute',
        duration: 4,
        wait: {A: 1},
        signal: {A: 2}
      },
      // Workload B: tokenize on CPU, fan out to local+remote, reduce, output.
      {
        id: 'b_tok',
        hardware: 'cpu',
        label: 'B:Tokenize',
        duration: 2,
        wait: {},
        signal: {B: 1}
      },
      {
        id: 'b_send',
        hardware: 'nic',
        label: 'B:Send',
        duration: 2,
        wait: {B: 1},
        signal: {R: 1}
      },
      {
        id: 'b_local',
        hardware: 'gpu0',
        label: 'B:Local',
        duration: 3,
        wait: {A: 2, B: 1},
        signal: {B: 2}
      },
      {
        id: 'b_remote',
        hardware: 'gpu1',
        label: 'B:Remote',
        duration: 3,
        wait: {R: 1},
        signal: {R: 2}
      },
      {
        id: 'b_reduce',
        hardware: 'gpu0',
        label: 'B:Reduce',
        duration: 2,
        wait: {B: 2, R: 2},
        signal: {B: 3}
      },
      {
        id: 'b_output',
        hardware: 'cpu',
        label: 'B:Output',
        duration: 1,
        wait: {B: 3},
        signal: {B: 4}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'Workload A\'s tokenization starts on the CPU. GPU 0, GPU 1, and the NIC ' +
            'are idle \u2014 no operations are ready for them yet.'
      },
      {
        tick: 2,
        text:
            'A:Tokenize completes, enabling A:Compute on GPU 0. B:Tokenize starts on ' +
            'the CPU (one operation at a time per hardware lane). Both workloads are ' +
            'progressing \u2014 A on GPU 0, B on CPU.'
      },
      {
        tick: 4,
        text:
            'B:Tokenize completes, signaling B to 1. B:Send starts on the NIC, ' +
            'transferring data to the remote machine. B:Local needs BOTH A \u2265 2 ' +
            '(A\'s compute result) AND B \u2265 1 \u2014 it can\'t start yet because A:Compute ' +
            'isn\'t done.'
      },
      {
        tick: 6,
        text:
            'A:Compute and B:Send both complete simultaneously. B:Local starts on ' +
            'GPU 0 \u2014 its frontier {A:2, B:2} merges knowledge from BOTH workloads ' +
            'through its semaphore waits. B:Remote starts on GPU 1 \u2014 it only knows ' +
            'about B\'s sent data ({B:1, R:2}), not A.'
      },
      {
        tick: 9,
        text:
            'B:Local and B:Remote both complete. B:Reduce starts, joining their ' +
            'frontiers: {A:2, B:3, R:2}. It carries transitive knowledge of ' +
            'A:Compute (via A:2) even though it never directly waited on A \u2014 the ' +
            'knowledge propagated through B:Local into semaphore B.'
      },
      {
        tick: 11,
        text:
            'B:Reduce completes. B:Output starts with frontier {A:2, B:4, R:2} \u2014 ' +
            'the full causal summary of the entire multi-domain computation. Any ' +
            'new workload C waiting on B \u2265 4 would instantly inherit this entire ' +
            'history from a single semaphore wait, without needing to know about ' +
            'semaphores A or R.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 8. Speculative Decoding Pipeline — draft/verify overlap
  // -------------------------------------------------------------------------
  {
    id: 'spec-decode',
    name: '8. Speculative Decoding',
    chain_through: ['K'],
    description:
        'A draft model generates speculative tokens on a small GPU while the main ' +
        'model verifies them on a large GPU. The key pipeline optimization: while ' +
        'Verify:R1 checks round 1, Draft:R2 speculatively starts round 2 \u2014 both ' +
        'GPUs overlap. Draft:R2 does NOT wait for verification (it depends on D:1, ' +
        'not V:1), so its frontier excludes V \u2014 the frontier precisely captures ' +
        'what is speculative vs verified.',
    hardware: [
      {id: 'main', label: 'Main GPU'},
      {id: 'draft', label: 'Draft GPU'},
    ],
    semaphores: [
      {id: 'K', label: 'KV Cache'},
      {id: 'D', label: 'Draft'},
      {id: 'V', label: 'Verify'},
    ],
    operations: [
      {
        id: 'prefill',
        hardware: 'main',
        label: 'Prefill',
        duration: 4,
        wait: {},
        signal: {K: 1}
      },
      {
        id: 'draft_r1',
        hardware: 'draft',
        label: 'Draft:R1',
        duration: 2,
        wait: {K: 1},
        signal: {D: 1}
      },
      {
        id: 'verify_r1',
        hardware: 'main',
        label: 'Verify:R1',
        duration: 2,
        wait: {D: 1},
        signal: {V: 1}
      },
      {
        id: 'draft_r2',
        hardware: 'draft',
        label: 'Draft:R2',
        duration: 2,
        wait: {D: 1},
        signal: {D: 2}
      },
      {
        id: 'verify_r2',
        hardware: 'main',
        label: 'Verify:R2',
        duration: 2,
        wait: {D: 2, V: 1},
        signal: {V: 2}
      },
      {
        id: 'output',
        hardware: 'main',
        label: 'Output',
        duration: 1,
        wait: {V: 2},
        signal: {K: 2}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'Prefill processes the prompt on the Main GPU, building the initial KV ' +
            'cache. The Draft GPU is idle \u2014 it cannot start drafting until the KV ' +
            'cache is ready.'
      },
      {
        tick: 4,
        text:
            'Prefill complete (K:1). Draft:R1 starts generating speculative tokens ' +
            'on the small Draft GPU. The Main GPU is idle during drafting \u2014 in real ' +
            'systems this gap is where the scheduler serves other requests.'
      },
      {
        tick: 6,
        text:
            'Draft:R1 complete (D:1). Now the pipeline overlap begins: Verify:R1 ' +
            'starts on the Main GPU AND Draft:R2 starts on the Draft GPU simultaneously. ' +
            'Draft:R2 waits on D \u2265 1 (its own previous round), NOT V \u2265 1 \u2014 it drafts ' +
            'speculatively without knowing if round 1 was accepted.'
      },
      {
        tick: 8,
        text:
            'Both complete. The frontier asymmetry reveals what is speculative: ' +
            'Draft:R2\'s frontier {K:1, D:2} has NO V component \u2014 it drafted without ' +
            'verification knowledge. Verify:R1\'s frontier {K:1, D:1, V:1} captures ' +
            'the full validated state. Verify:R2 waits on BOTH D:2 and V:1, merging ' +
            'the speculative and verified branches.'
      },
      {
        tick: 10,
        text:
            'Verify:R2 complete. Its frontier {K:1, D:2, V:2} includes everything: ' +
            'KV cache, both draft rounds, and both verification rounds. If round 1 ' +
            'had been rejected, the system would have discarded Draft:R2\'s work and ' +
            're-drafted from the verified position \u2014 the speculative frontier ' +
            'would never have propagated further.'
      },
      {
        tick: 11,
        text:
            'Output done. Final frontier {K:2, D:2, V:2} \u2014 everything is verified ' +
            'and the KV cache is updated. Downstream requests waiting on K \u2265 2 ' +
            'inherit the complete speculative decoding history through a single ' +
            'semaphore wait.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 9. KV Cache Paging — NVMe memory management with request interleaving
  // -------------------------------------------------------------------------
  {
    id: 'kv-paging',
    name: '9. KV Cache Paging',
    description:
        'Two requests share a GPU with limited memory. When request A pauses, its ' +
        'KV cache is evicted to NVMe. But B:Prefill cannot start until eviction ' +
        'completes \u2014 there is not enough GPU memory for both. Similarly, A\'s ' +
        'cache cannot be loaded back until B:Decode finishes and releases memory. ' +
        'These memory-pressure dependencies create cross-request causal links ' +
        'through the page manager, visible in the frontiers.',
    hardware: [
      {id: 'gpu', label: 'GPU', count: 1},
      {id: 'nvme', label: 'NVMe'},
    ],
    semaphores: [
      {id: 'A', label: 'Request A'},
      {id: 'B', label: 'Request B'},
      {id: 'P', label: 'Page State'},
    ],
    operations: [
      {
        id: 'a_prefill',
        hardware: 'gpu',
        label: 'A:Prefill',
        duration: 2,
        wait: {},
        signal: {A: 1}
      },
      {
        id: 'a_decode',
        hardware: 'gpu',
        label: 'A:Decode',
        duration: 2,
        wait: {A: 1},
        signal: {A: 2}
      },
      {
        id: 'evict',
        hardware: 'nvme',
        label: 'Evict',
        duration: 2,
        wait: {A: 2},
        signal: {P: 1}
      },
      {
        id: 'b_prefill',
        hardware: 'gpu',
        label: 'B:Prefill',
        duration: 2,
        wait: {P: 1},
        signal: {B: 1}
      },
      {
        id: 'b_decode',
        hardware: 'gpu',
        label: 'B:Decode',
        duration: 2,
        wait: {B: 1},
        signal: {B: 2}
      },
      {
        id: 'load',
        hardware: 'nvme',
        label: 'Load',
        duration: 2,
        wait: {P: 1, B: 2},
        signal: {P: 2}
      },
      {
        id: 'a_resume',
        hardware: 'gpu',
        label: 'A:Resume',
        duration: 2,
        wait: {P: 2},
        signal: {A: 3}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'Request A starts prefilling, consuming GPU memory for its KV cache. ' +
            'B is ready (no semaphore waits in the base graph) but must wait for ' +
            'the GPU \u2014 hardware scheduling, not causal ordering.'
      },
      {
        tick: 2,
        text:
            'A:Prefill done. A:Decode starts (definition order tiebreak). B is still ' +
            'waiting for the GPU. In the previous version of this scenario (without ' +
            'memory pressure), B would start here alongside Evict. Now it must wait.'
      },
      {
        tick: 4,
        text:
            'A:Decode done. Evict starts on NVMe, saving A\'s KV cache to disk. ' +
            'B:Prefill cannot start \u2014 it waits on P \u2265 1 (eviction complete) because ' +
            'there is not enough GPU memory for both. The GPU sits idle while the ' +
            'NVMe works. Memory pressure serializes what could have been parallel.'
      },
      {
        tick: 6,
        text:
            'Eviction complete (P:1). B:Prefill starts \u2014 GPU memory is free. Its ' +
            'frontier {A:2, P:1, B:1} carries knowledge of A\'s work through the ' +
            'eviction dependency. This is not scheduling: B causally depends on A\'s ' +
            'eviction completing.'
      },
      {
        tick: 10,
        text:
            'B:Decode complete (B:2). Load starts on NVMe \u2014 it waits on BOTH ' +
            'P \u2265 1 (knows where A\'s data is) AND B \u2265 2 (B released GPU memory). ' +
            'Load\'s frontier {A:2, B:2, P:2} merges both requests: A\'s compute ' +
            'and B\'s intervening work flow into the page manager.'
      },
      {
        tick: 12,
        text:
            'Load complete. A:Resume starts with frontier {A:3, B:2, P:2} \u2014 it ' +
            'includes B:2! Unlike the independent-streams scenario, memory pressure ' +
            'created real cross-request causal links. A:Resume knows about B because ' +
            'A\'s cache could not be reloaded until B released memory.'
      },
      {
        tick: 14,
        text:
            'All done. A: {A:3, B:2, P:2}. B: {A:2, B:2, P:1}. Neither is disjoint ' +
            '\u2014 memory pressure coupled them through P. Compare with scenario 4 ' +
            '(Independent Streams) where hardware sharing created zero causal leakage. ' +
            'With per-block semaphores, only the specific blocks that page in/out would ' +
            'appear in each frontier, giving finer-grained causal tracking.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 10. Multi-Modal Remote Inference — text + vision across network boundary
  // -------------------------------------------------------------------------
  {
    id: 'multi-modal',
    name: '10. Multi-Modal Remote',
    chain_through: ['F'],
    description:
        'A multi-modal request processes text locally and images remotely. Text ' +
        'tokenization and image upload start in parallel on different hardware. The ' +
        'text and vision pipelines have completely disjoint frontier dimensions ' +
        'until CrossAttn fuses them \u2014 the multi-modal join point. Network hops ' +
        'appear in the frontier (N:1) so downstream consumers know the image ' +
        'traversed the network.',
    hardware: [
      {id: 'cpu', label: 'CPU'},
      {id: 'local', label: 'Local GPU'},
      {id: 'nic', label: 'NIC'},
      {id: 'remote', label: 'Remote GPU'},
    ],
    semaphores: [
      {id: 'T', label: 'Text'},
      {id: 'N', label: 'Network'},
      {id: 'V', label: 'Vision'},
      {id: 'F', label: 'Fusion'},
    ],
    operations: [
      {
        id: 'tokenize',
        hardware: 'cpu',
        label: 'Tokenize',
        duration: 1,
        wait: {},
        signal: {T: 1}
      },
      {
        id: 'upload',
        hardware: 'nic',
        label: 'Upload',
        duration: 1,
        wait: {},
        signal: {N: 1}
      },
      {
        id: 'embed',
        hardware: 'local',
        label: 'Embed',
        duration: 3,
        wait: {T: 1},
        signal: {T: 2}
      },
      {
        id: 'vit',
        hardware: 'remote',
        label: 'ViT',
        duration: 3,
        wait: {N: 1},
        signal: {V: 1}
      },
      {
        id: 'download',
        hardware: 'nic',
        label: 'Download',
        duration: 1,
        wait: {V: 1},
        signal: {V: 2}
      },
      {
        id: 'cross_attn',
        hardware: 'local',
        label: 'CrossAttn',
        duration: 2,
        wait: {T: 2, V: 2},
        signal: {F: 1}
      },
      {
        id: 'generate',
        hardware: 'local',
        label: 'Generate',
        duration: 2,
        wait: {F: 1},
        signal: {F: 2}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'Both modalities start processing in parallel: Tokenize on CPU handles ' +
            'text, Upload on NIC sends the image to the remote machine. Local and ' +
            'Remote GPUs are idle, waiting for their respective inputs.'
      },
      {
        tick: 1,
        text:
            'Both transfers complete. Embed starts on the Local GPU (text embedding) ' +
            'and ViT starts on the Remote GPU (vision encoding). The two pipelines ' +
            'have completely disjoint frontier dimensions: Embed tracks T, ViT tracks ' +
            'N and V. No causal connection between them yet.'
      },
      {
        tick: 4,
        text:
            'Both Embed and ViT complete simultaneously. The text pipeline is done, ' +
            'but CrossAttn needs vision features too. Download starts, transferring ' +
            'ViT\'s output back over the network. The Local GPU waits for one tick.'
      },
      {
        tick: 5,
        text:
            'Download complete (V:2). CrossAttn starts \u2014 the multi-modal fusion ' +
            'point. Its frontier {T:2, N:1, V:2, F:1} merges both modalities for the ' +
            'first time. The network hop appears as N:1: any downstream consumer can ' +
            'see that the image was processed remotely.'
      },
      {
        tick: 7,
        text:
            'CrossAttn done. Generate starts with frontier {T:2, N:1, V:2, F:2} \u2014 ' +
            'the complete multi-modal summary. A downstream scheduler waiting on ' +
            'F \u2265 2 inherits knowledge of text processing, network transfer, and ' +
            'vision encoding through a single semaphore wait on F, without needing ' +
            'to know about semaphores T, N, or V.'
      },
      {
        tick: 9,
        text:
            'All done. The final frontier captures the full inference path across ' +
            'four hardware resources. This is the pattern for any multi-modal ' +
            'serving system: independent modality pipelines with disjoint frontiers ' +
            'converging at a fusion point that produces the combined knowledge.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 11. Voice-to-Chat Pipeline — cross-device pipeline with DMA transfer
  // -------------------------------------------------------------------------
  {
    id: 'voice-chat',
    name: '11. Voice-to-Chat Pipeline',
    description:
        'Cross-device pipeline from voice input to LLM response on a laptop. ' +
        'Audio is preprocessed on the CPU, transcribed by Whisper on the APU, ' +
        'transferred via PCIe DMA to VRAM, then processed by Phi-3 on the dGPU. ' +
        'A background embedding model runs concurrently on the NPU with a ' +
        'completely disjoint frontier axis.',
    hardware: [
      {id: 'cpu', label: 'CPU'},
      {id: 'apu', label: 'APU'},
      {id: 'pcie', label: 'PCIe DMA'},
      {id: 'dgpu', label: 'dGPU'},
      {id: 'npu', label: 'NPU'},
    ],
    semaphores: [
      {id: 'A', label: 'Audio'},
      {id: 'W', label: 'Whisper'},
      {id: 'D', label: 'DMA'},
      {id: 'P', label: 'Phi'},
      {id: 'E', label: 'Embed'},
    ],
    operations: [
      {
        id: 'audio',
        hardware: 'cpu',
        label: 'Audio Prep',
        duration: 1,
        wait: {},
        signal: {A: 1}
      },
      {
        id: 'whisper',
        hardware: 'apu',
        label: 'Whisper',
        duration: 3,
        wait: {A: 1},
        signal: {W: 1}
      },
      {
        id: 'dma',
        hardware: 'pcie',
        label: 'DMA Transfer',
        duration: 1,
        wait: {W: 1},
        signal: {D: 1}
      },
      {
        id: 'prefill',
        hardware: 'dgpu',
        label: 'Phi-3 Prefill',
        duration: 2,
        wait: {D: 1},
        signal: {P: 1}
      },
      {
        id: 'decode',
        hardware: 'dgpu',
        label: 'Phi-3 Decode',
        duration: 3,
        wait: {P: 1},
        signal: {P: 2}
      },
      {
        id: 'embed',
        hardware: 'npu',
        label: 'BGE Embed',
        duration: 4,
        wait: {},
        signal: {E: 1}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'The voice pipeline and background embedding start simultaneously. ' +
            'Audio Prep runs on CPU while BGE Embed runs on the NPU \u2014 different ' +
            'devices, different semaphore axes, zero causal relationship. The entire ' +
            'voice pipeline was submitted atomically before any work began.'
      },
      {
        tick: 1,
        text:
            'Audio preprocessing completes. Whisper begins on the APU \u2014 it was ' +
            'queued on the APU\u2019s hardware FIFO at submission time, waiting for ' +
            'the Audio semaphore. No round-trip to the application between stages.'
      },
      {
        tick: 4,
        text:
            'Whisper completes and DMA begins transferring token IDs from system ' +
            'memory to VRAM. BGE Embed also finishes this tick with frontier {E:1} ' +
            '\u2014 completely disjoint from the voice pipeline. The two workloads ' +
            'shared the machine but created zero causal interference.'
      },
      {
        tick: 5,
        text: 'DMA completes. Phi-3 Prefill starts on the dGPU with frontier ' +
            '{A:1, W:1, D:1, P:1} \u2014 it inherits knowledge of CPU audio ' +
            'processing, APU transcription, and PCIe transfer through the ' +
            'semaphore chain. No direct interaction with those devices was needed.'
      },
      {
        tick: 7,
        text:
            'Prefill completes, feeding the KV cache into autoregressive decoding.'
      },
      {
        tick: 10,
        text:
            'Phi-3 Decode completes. Final frontier: {A:1, W:1, D:1, P:2}. This ' +
            'compact vector captures the complete provenance \u2014 audio processing ' +
            'on CPU, transcription on APU, DMA to VRAM, and two-phase generation on ' +
            'dGPU. Any downstream consumer (TTS, logging) inherits this full history ' +
            'through a single semaphore wait.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 12. RAG Pipeline — heterogeneous device placement by weight residency
  // -------------------------------------------------------------------------
  {
    id: 'rag-pipeline',
    name: '12. RAG Pipeline',
    chain_through: ['G'],
    description:
        'Retrieval-augmented generation spanning CPU and two GPUs. Work bounces ' +
        'between devices based on weight residency: BGE embedding on GPU 0, ' +
        'vector search on CPU, reranking on GPU 1, then generation back on GPU 0. ' +
        'The final answer\u2019s frontier captures axes from all three compute ' +
        'domains \u2014 complete provenance without explicit dependency tracking.',
    hardware: [
      {id: 'cpu', label: 'CPU'},
      {id: 'gpu0', label: 'GPU 0'},
      {id: 'gpu1', label: 'GPU 1'},
    ],
    semaphores: [
      {id: 'T', label: 'Token'},
      {id: 'B', label: 'BGE'},
      {id: 'V', label: 'Vector'},
      {id: 'K', label: 'Rerank'},
      {id: 'G', label: 'Generate'},
    ],
    operations: [
      {
        id: 'tokenize',
        hardware: 'cpu',
        label: 'Tokenize',
        duration: 1,
        wait: {},
        signal: {T: 1}
      },
      {
        id: 'embed',
        hardware: 'gpu0',
        label: 'BGE Embed',
        duration: 2,
        wait: {T: 1},
        signal: {B: 1}
      },
      {
        id: 'search',
        hardware: 'cpu',
        label: 'Vector Search',
        duration: 2,
        wait: {B: 1},
        signal: {V: 1}
      },
      {
        id: 'rerank',
        hardware: 'gpu1',
        label: 'Rerank',
        duration: 2,
        wait: {V: 1},
        signal: {K: 1}
      },
      {
        id: 'generate',
        hardware: 'gpu0',
        label: 'Generate',
        duration: 3,
        wait: {K: 1},
        signal: {G: 1}
      },
    ],
    annotations: [
      {
        tick: 0,
        text: 'A RAG query arrives. Tokenization runs on the CPU. The entire ' +
            'pipeline \u2014 spanning CPU, GPU 0, CPU again, GPU 1, and back to ' +
            'GPU 0 \u2014 was submitted atomically. Each stage is placed on the ' +
            'device where the relevant weights are resident.'
      },
      {
        tick: 1,
        text:
            'Tokenization completes. BGE Embed starts on GPU 0 (BGE weights ' +
            'resident there). The CPU is now idle \u2014 it will be needed again ' +
            'for vector search after embedding completes.'
      },
      {
        tick: 3,
        text:
            'Embedding completes. Vector Search starts on CPU (the FAISS index ' +
            'lives in system memory). Both GPUs are idle: GPU 0 finished embedding, ' +
            'GPU 1 is waiting for search results to rerank.'
      },
      {
        tick: 5,
        text:
            'Vector Search completes. Rerank starts on GPU 1 (cross-encoder weights ' +
            'resident there). Its frontier {T:1, B:1, V:1, K:1} carries knowledge ' +
            'of tokenization, embedding, AND vector search transitively \u2014 GPU 1 ' +
            'never interacted with GPU 0 directly.'
      },
      {
        tick: 7,
        text:
            'Reranking completes. Generate starts back on GPU 0 (Llama weights). ' +
            'Work has bounced: GPU 0 \u2192 CPU \u2192 GPU 1 \u2192 GPU 0. GPU 0 ' +
            'now knows about GPU 1\u2019s reranking through the semaphore chain.'
      },
      {
        tick: 10,
        text:
            'Generation completes. Final frontier: {T:1, B:1, V:1, K:1, G:1}. ' +
            'Every stage that contributed to the answer is represented. Any ' +
            'downstream consumer inherits this complete provenance through a ' +
            'single wait on G\u22651.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 13. Multi-Model Orchestration — wide DAG across 7 models
  // -------------------------------------------------------------------------
  {
    id: 'multi-model',
    name: '13. Multi-Model Orchestration',
    description:
        'A multimedia request touches 7 models: voice and image are understood ' +
        'in parallel, merged at a language model, then forked to code generation ' +
        'and voice synthesis. The final response\u2019s frontier captures the ' +
        'complete provenance of every model that contributed. Use the GPU slider ' +
        'to see how parallelism in the understanding and output phases changes ' +
        'the schedule.',
    hardware: [
      {id: 'cpu', label: 'CPU'},
      {id: 'gpu', label: 'GPU', count: 2},
    ],
    semaphores: [
      {id: 'A', label: 'Audio'},
      {id: 'I', label: 'Image'},
      {id: 'W', label: 'Whisper'},
      {id: 'V', label: 'Vision'},
      {id: 'L', label: 'Llama'},
      {id: 'K', label: 'Code'},
      {id: 'T', label: 'TTS'},
      {id: 'S', label: 'SDXL'},
      {id: 'R', label: 'Response'},
    ],
    operations: [
      {
        id: 'audio_in',
        hardware: 'cpu',
        label: 'Audio Prep',
        duration: 1,
        wait: {},
        signal: {A: 1}
      },
      {
        id: 'image_in',
        hardware: 'cpu',
        label: 'Image Prep',
        duration: 1,
        wait: {},
        signal: {I: 1}
      },
      {
        id: 'whisper',
        hardware: 'gpu',
        label: 'Whisper',
        duration: 2,
        wait: {A: 1},
        signal: {W: 1}
      },
      {
        id: 'clip',
        hardware: 'gpu',
        label: 'CLIP',
        duration: 2,
        wait: {I: 1},
        signal: {V: 1}
      },
      {
        id: 'llama',
        hardware: 'gpu',
        label: 'Llama-70B',
        duration: 3,
        wait: {W: 1, V: 1},
        signal: {L: 1}
      },
      {
        id: 'code_llama',
        hardware: 'gpu',
        label: 'Code-Llama',
        duration: 2,
        wait: {L: 1},
        signal: {K: 1}
      },
      {
        id: 'tts',
        hardware: 'gpu',
        label: 'TTS',
        duration: 2,
        wait: {L: 1},
        signal: {T: 1}
      },
      {
        id: 'sdxl',
        hardware: 'gpu',
        label: 'SDXL',
        duration: 2,
        wait: {K: 1},
        signal: {S: 1}
      },
      {
        id: 'assemble',
        hardware: 'cpu',
        label: 'Assemble',
        duration: 1,
        wait: {K: 1, S: 1, T: 1},
        signal: {R: 1}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'A multimedia request arrives: voice note + image. Audio Prep starts ' +
            'on CPU. The entire 9-operation pipeline \u2014 spanning 7 models ' +
            'across CPU and GPUs \u2014 was submitted atomically.'
      },
      {
        tick: 2,
        text:
            'Image Prep done. CLIP Vision starts on a second GPU, running in ' +
            'parallel with Whisper on the first. Two understanding models working ' +
            'simultaneously, each with independent semaphore axes.'
      },
      {
        tick: 4,
        text:
            'Both Whisper and CLIP complete. Llama-70B starts \u2014 the merge ' +
            'point. It waits on BOTH W\u22651 AND V\u22651, inheriting frontiers ' +
            'from the audio and image branches. Its frontier {A:1, I:1, W:1, ' +
            'V:1, L:1} captures everything upstream.'
      },
      {
        tick: 7,
        text: 'Llama-70B completes. Two models fork in parallel: Code-Llama ' +
            'generates code while TTS synthesizes voice. Both inherit Llama\u2019s ' +
            'full frontier, but their output axes (K vs T) are disjoint.'
      },
      {
        tick: 9,
        text:
            'Code-Llama and TTS both complete. SDXL starts generating an image ' +
            'from the code description. Try the GPU slider at 1: without parallel ' +
            'GPU lanes, Whisper/CLIP and Code-Llama/TTS must serialize, adding ' +
            '3 ticks.'
      },
      {
        tick: 12,
        text:
            'Assemble completes with frontier {A:1, I:1, W:1, V:1, L:1, K:1, ' +
            'S:1, T:1, R:1} \u2014 nine entries capturing complete provenance: ' +
            'audio prep, image prep, speech-to-text, image understanding, language ' +
            'modeling, code generation, voice synthesis, image synthesis, and ' +
            'assembly.'
      },
    ],
  },

  // -------------------------------------------------------------------------
  // 14. Streaming Translation — pipelined voice-to-voice across speech chunks
  // -------------------------------------------------------------------------
  {
    id: 'streaming-translation',
    name: '14. Streaming Translation',
    chain_through: ['S'],
    iteration_deps: {
      'decode': ['S'],
      'synthesize': ['T'],
    },
    description:
        'Live voice-to-voice translation: audio chunks flow through VAD, encoding, ' +
        'decoding (STT), LLM translation, TTS synthesis, and playback. Each depth ' +
        'iteration is one speech chunk. Two inter-iteration state dependencies ' +
        'create "skip" connections in the DAG: the decoder carries beam search ' +
        'state (S axis), and TTS carries prosody state (T axis). These skips ' +
        'propagate frontiers directly between the same stage across chunks, ' +
        'bypassing the normal pipeline chain. Try depth \u2265 2 with 2 GPUs.',
    hardware: [
      {id: 'cpu', label: 'CPU'},
      {id: 'gpu', label: 'GPU', count: 1},
    ],
    semaphores: [
      {id: 'D', label: 'VAD'},
      {id: 'E', label: 'Encoder'},
      {id: 'S', label: 'STT'},
      {id: 'L', label: 'LLM'},
      {id: 'T', label: 'TTS'},
      {id: 'P', label: 'Playback'},
    ],
    operations: [
      {
        id: 'vad',
        hardware: 'cpu',
        label: 'VAD',
        duration: 1,
        wait: {},
        signal: {D: 1}
      },
      {
        id: 'encode',
        hardware: 'gpu',
        label: 'Encode',
        duration: 2,
        wait: {D: 1},
        signal: {E: 1}
      },
      {
        id: 'decode',
        hardware: 'gpu',
        label: 'Decode',
        duration: 2,
        wait: {E: 1},
        signal: {S: 1}
      },
      {
        id: 'translate',
        hardware: 'gpu',
        label: 'Translate',
        duration: 3,
        wait: {S: 1},
        signal: {L: 1}
      },
      {
        id: 'synthesize',
        hardware: 'gpu',
        label: 'Synthesize',
        duration: 2,
        wait: {L: 1},
        signal: {T: 1}
      },
      {
        id: 'playback',
        hardware: 'cpu',
        label: 'Playback',
        duration: 1,
        wait: {T: 1},
        signal: {P: 1}
      },
    ],
    annotations: [
      {
        tick: 0,
        text:
            'A speech chunk arrives. VAD (voice activity detection) runs on the CPU, ' +
            'segmenting the audio stream. The entire 6-stage pipeline was submitted ' +
            'atomically \u2014 all downstream stages are queued on their hardware FIFOs, ' +
            'waiting for semaphore signals to propagate.'
      },
      {
        tick: 1,
        text:
            'Encoder starts on the GPU, converting audio features to a latent ' +
            'representation. With one GPU, every stage serializes through the single ' +
            'lane. Try depth \u2265 2 with 2 GPUs to see chunks at different pipeline ' +
            'stages overlap, and the inter-iteration "skip" arrows on Decode and ' +
            'Synthesize.'
      },
      {
        tick: 3,
        text:
            'Decoder (STT) starts, producing source-language tokens from the encoder ' +
            'output. The decoder is stateful: it carries beam search hypotheses and ' +
            'hidden state from the previous chunk. At depth > 1, this creates a ' +
            '"skip" dependency \u2014 Decode[N+1] waits directly on Decode[N]\u2019s ' +
            'STT semaphore, propagating decoder state through the frontier.'
      },
      {
        tick: 5,
        text:
            'LLM translation begins \u2014 the pipeline bottleneck at 3 ticks. It ' +
            'converts source tokens to target-language tokens. In a streaming system ' +
            'like a babelfish, partial LLM output could feed TTS incrementally; here ' +
            'each stage completes fully before the next begins.'
      },
      {
        tick: 8,
        text:
            'TTS synthesis converts target-language tokens to audio waveforms. Like ' +
            'the decoder, TTS carries state across chunks: prosody, pitch contour, ' +
            'and speaking rate must be continuous. At depth > 1, Synthesize[N+1] ' +
            'waits on Synthesize[N]\u2019s TTS semaphore \u2014 a "skip" that bypasses ' +
            'the normal pipeline chain entirely.'
      },
      {
        tick: 10,
        text:
            'Playback starts delivering translated audio to the speaker. End-to-end ' +
            'latency for one chunk: 11 ticks. At depth \u2265 2, the DAG shows the two ' +
            'skip connections: Decode\u2192Decode and Synthesize\u2192Synthesize ' +
            'crossing iteration boundaries while the rest of the pipeline flows ' +
            'normally.'
      },
    ],
  },

];
