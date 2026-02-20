# Multi-Model Scheduling Scenarios

This document walks through concrete multi-model workloads on three hardware
configurations, showing how the frontier system enables efficient scheduling,
memory sharing, and pipelining across heterogeneous devices.

Each scenario focuses on the interactions that make multi-model serving
difficult: memory contention, cross-model data flow, interleaving on shared
hardware, and priority-driven scheduling. The
[interactive visualizer](visualizer/) demonstrates the
underlying frontier mechanics; these scenarios show how those mechanics apply
to realistic deployments.

---

## Hardware Configuration A: High-End Laptop

### Hardware

- 1 CPU (8P+4E cores, integrated NPU)
- 1 APU / integrated GPU (shares system memory, 4 CUs)
- 1 discrete GPU (8GB VRAM, PCIe 4.0)

### Models (3 concurrent)

| Model | Type | Size | Placement | Frequency |
|-------|------|------|-----------|-----------|
| Phi-3-mini-4k | LLM | 2.3GB | dGPU | Continuous (chat) |
| Whisper-small | STT | 0.5GB | APU or CPU | Bursty (voice input) |
| MiniLM-L6 | Embedding | 80MB | CPU / NPU | Batch (local search index) |

Total GPU memory: 2.3GB + 0.5GB = 2.8GB of 8GB, leaving headroom for KV
cache and activations.

### Key Interactions

**Voice-to-chat pipeline**: The user speaks, Whisper transcribes, Phi-3
generates a response. This is a cross-device pipeline:

```text
CPU: audio preprocessing
  signal(audio_sem, N)
  frontier: {cpu_axis: N}

APU: Whisper encoder + decoder
  wait(audio_sem, N), signal(whisper_sem, N)
  frontier: {cpu_axis: N, apu_axis: M}

dGPU: Phi-3 prefill + decode
  wait(whisper_sem, N), signal(phi_sem, T)
  frontier: {cpu_axis: N, apu_axis: M, dgpu_axis: K}
```

The pipeline is submitted atomically. The dGPU's work is queued before
Whisper has started — the hardware FIFO on the dGPU holds the work until
the semaphore signal arrives from the APU. No round-trip to the application
between Whisper completing and Phi-3 starting.

On this hardware, the APU and dGPU are on different buses (the APU accesses
system memory directly, the dGPU uses PCIe). Whisper's output must be
transferred from system memory to VRAM. The frontier captures the transfer
as part of the chain — the DMA engine has its own queue axis, and the dGPU's
wait includes the DMA completion transitively. The
[visualizer's scenario 11](visualizer/?scenario=voice-chat)
shows this pipeline, including the DMA transfer as a distinct hardware lane
and the concurrent background embedding on the NPU.

**Background embedding**: While the chat is idle (user reading a response),
MiniLM runs on the CPU or NPU, indexing local documents. Its semaphore axis
is entirely disjoint from the chat pipeline — zero frontier interference
despite sharing the CPU. If a new voice input arrives, the CPU preempts
MiniLM (priority scheduling) and processes the audio, then resumes MiniLM
with no synchronization impact on the chat pipeline.

**Memory sharing**: Phi-3's KV cache and Whisper's activations share the
dGPU's 8GB. When Whisper finishes and the user is reading the response,
Whisper's activation buffers' death frontiers allow the allocator to reuse
that memory for Phi-3's growing KV cache. When a new voice input arrives,
Whisper's weights (still resident) need activation space again — the
allocator checks death frontiers on the KV cache blocks and reuses any
that have been fully consumed by Phi-3's decode steps.

### What this demonstrates

Even on a laptop with one dGPU, the frontier system eliminates round-trips
in the voice-to-chat pipeline, enables zero-cost interleaving of background
embedding work, and provides allocation-free buffer sharing between models
through death frontier tracking. The alternative — three separate processes
with IPC for cross-model communication — would add milliseconds of latency
to the voice-to-chat pipeline that is targeting sub-second total latency.

---

## Hardware Configuration B: High-End Workstation

### Hardware

- 2 CPUs (32 cores each, 2 NUMA domains)
- 2 discrete GPUs (24GB VRAM each, PCIe 5.0, no P2P)
- 2 NVMe SSDs (7GB/s read each, striped)

### Models (8 concurrent)

| # | Model | Type | Size | GPU | Notes |
|---|-------|------|------|-----|-------|
| 1 | Llama-3-8B | LLM | 5GB | GPU 0 | Primary chat |
| 2 | Llama-3-8B | LLM (draft) | 1GB (quantized) | GPU 0 | Speculative decoding draft |
| 3 | Whisper-large-v3 | STT | 2GB | GPU 1 | Voice input |
| 4 | Bark | TTS | 4GB | GPU 1 | Voice output |
| 5 | SDXL | Diffusion | 6GB | GPU 0 or 1 | Image generation |
| 6 | BGE-large | Embedding | 0.5GB | Either | RAG indexing |
| 7 | Reranker | Cross-encoder | 1GB | Either | RAG reranking |
| 8 | CLIP-ViT-L | Vision | 1GB | Either | Image understanding |

Total GPU memory needed: ~20GB across two 24GB GPUs. This fits, but with
limited room for KV caches, activations, and SDXL's intermediate latents.

### Scenario B1: Speculative Decoding with Concurrent STT/TTS

The user is in a voice conversation with the LLM. Whisper transcribes on
GPU 1, Llama-3-8B generates on GPU 0 with speculative decoding, and Bark
synthesizes speech on GPU 1.

```text
Voice-to-text:
  CPU NUMA1: audio preprocessing → signal(audio_sem, N)
  GPU 1: Whisper encode/decode → signal(whisper_sem, N)
  Transfer: GPU 1 → host → GPU 0 (token IDs, small)

Text generation (speculative):
  GPU 0: Llama draft model generates 4 tokens → signal(draft_sem, R)
  GPU 0: Llama main model verifies → signal(verify_sem, R)
  GPU 0: Llama draft round 2 (overlaps verify) → signal(draft_sem, R+1)
    draft frontier: {dgpu0_axis: ..., draft_axis: R+1}  (no verify component)
  GPU 0: Llama main verify round 2 → signal(verify_sem, R+1)
    verify frontier: {dgpu0_axis: ..., draft_axis: R+1, verify_axis: R+1}

Text-to-speech:
  Transfer: GPU 0 → host → GPU 1 (generated tokens)
  GPU 1: Bark synthesis → signal(tts_sem, N)
  CPU NUMA1: audio output
```

The two GPUs operate in a pipeline. GPU 0 and GPU 1 overlap: while GPU 0
generates the next tokens, GPU 1 synthesizes the previous tokens. The
semaphore chain carries the full causal history — the TTS output's frontier
includes the Whisper, Llama, and Bark axes transitively.

Within GPU 0, speculative decoding overlaps draft and verify rounds. The
draft model's frontier structurally lacks the verify axis, encoding what
is speculative. If verification rejects draft tokens, the scheduler knows
exactly which frontier entries are invalidated.

During the generation phase, GPU 1 has idle compute capacity (Whisper
finished, Bark hasn't started). The scheduler fills this gap with BGE
embedding work — the BGE axis is disjoint from the voice pipeline, so
interleaving is safe.

### Scenario B2: SDXL Under Memory Pressure

A user requests image generation while the LLM is active. SDXL needs ~6GB
for weights + UNet latents. GPU 0 has Llama (5GB weights + KV cache) and
the draft model (1GB). GPU 1 has Whisper (2GB) + Bark (4GB). Neither GPU
has 6GB free.

#### Step 1: Deallocate cold model weights

The scheduler decides to drop Bark's weights (lowest priority, longest
expected idle time). Weights are read-only — they don't need to be saved
anywhere, just deallocated once all in-flight compute using them has
completed. The original weights remain available on disk or in system
memory for future reload.

```text
Deallocate Bark weights (4GB) on GPU 1:
  death_frontier: {bark_axis: latest_bark_epoch, dgpu1_axis: ...}
  Buffers return to GPU 1's free pool

GPU 1: load SDXL weights from system memory or disk
  try_reuse(sdxl_queue, freed_buffers)
  dominates(sdxl_queue_frontier, bark_death_frontier)?
    Yes — Bark's last compute completed before dealloc, which happened
    before SDXL's current epoch. Reuse is safe.
  signal(sdxl_sem, 1)
```

No DMA "eviction" step — Bark's weights are immutable, so dropping them
is just a deallocation. The death frontier on the freed buffers ensures
SDXL cannot reuse the memory until Bark's compute has actually completed.

#### Step 2: SDXL generation

```text
GPU 1: SDXL UNet denoising (50 steps)
  Each step: wait(sdxl_sem, S), signal(sdxl_sem, S+1)
  frontier: {dgpu1_axis: ...}
```

#### Step 3: Bark weight reload

When SDXL completes and the user sends another voice message:

```text
SDXL buffers deallocated:
  death_frontier: {dgpu1_axis: sdxl_final_epoch}

GPU 1: reload Bark weights from system memory or disk
  try_reuse(bark_queue, freed_sdxl_buffers) — dominance check
  signal(bark_sem, next)

GPU 1: Bark synthesis
  wait(bark_sem, next)
```

Weight load/deallocation participates in the same frontier system as
compute — no special-purpose memory management synchronization.

### Scenario B2b: KV Cache Paging to NVMe

Unlike weights, KV cache blocks are mutable — they contain computed state
that would be lost if simply deallocated. When the system runs many
concurrent chat sessions, the total KV cache across all sessions can exceed
GPU memory. The scheduler pages cold KV cache blocks to host memory or
NVMe.

```text
Llama sessions 1-10 active on GPU 0. Session 3 goes idle (user reading).

Page manager evicts session 3's KV blocks:
  NVMe DMA: write KV blocks from GPU 0 to SSD
    wait(llama_sem, session_3_latest_epoch)  // ensure decode finished
    signal(page_sem, P)
    frontier: {llama_axis: session_3_epoch, nvme_axis: P}

  KV block buffers freed → death_frontier recorded
  Other sessions can reuse that GPU memory for their growing KV caches.

Session 3 resumes (user sends a new message):
  NVMe DMA: read KV blocks from SSD back to GPU 0
    wait(page_sem, P)      // blocks are on SSD at this location
    signal(page_sem, P+1)
    frontier: {nvme_axis: P+1, ...}

  GPU 0: continue decoding for session 3
    wait(page_sem, P+1)   // KV cache restored
    signal(llama_sem, session_3_next_epoch)
```

The KV cache paging path has genuine NVMe axis involvement — the data
must be written to and read from the SSD. The frontier captures this:
when session 3 resumes, its frontier includes the NVMe axis, documenting
that the KV cache traversed persistent storage. Any downstream consumer
of session 3's output inherits this information transitively.

### Scenario B2c: Fine-Tuning with Checkpoint Paging

Fine-tuning is the case where weights *are* mutable and paging to NVMe
applies directly. During LoRA fine-tuning of the 8B model on GPU 0, the
optimizer state (momentum, variance) and gradient accumulation buffers
can exceed available memory:

```text
GPU 0: forward pass (weights + activations)
  signal(train_sem, step*3 + 1)

GPU 0: backward pass (gradients)
  signal(train_sem, step*3 + 2)

NVMe DMA: page out optimizer state from previous step to SSD
  (freeing memory for the current step's gradient accumulation)
  wait(train_sem, step*3 + 1)  // forward done, old optimizer state unused
  signal(page_sem, P)

GPU 0: optimizer step (needs gradients + paged-in optimizer state)
  wait(train_sem, step*3 + 2)  // backward done
  wait(page_sem, P+1)          // optimizer state loaded back
  signal(train_sem, (step+1)*3)

NVMe DMA: page in optimizer state for next step
  wait(page_sem, P)
  signal(page_sem, P+1)
```

The training loop pipelines forward/backward passes with optimizer state
paging. The NVMe DMA axis interleaves with the compute axis, and the
frontier tracks which training step each operation belongs to. Checkpoint
writes to NVMe are just additional signals on the page semaphore —
they compose with the rest of the training pipeline through the same
frontier algebra.

### Scenario B3: RAG Pipeline with Heterogeneous Placement

A RAG query arrives: embed the query, retrieve documents, rerank, and
generate an answer. The scheduler places work based on where weights are
resident and which GPU has capacity:

```text
CPU NUMA0: tokenize query → signal(rag_sem, 1)

GPU 0 (BGE weights resident): embed query
  wait(rag_sem, 1), signal(rag_sem, 2)
  frontier: {cpu0_axis: ..., dgpu0_axis: ...}

CPU NUMA0: vector search (retrieval from index) → signal(rag_sem, 3)

GPU 1 (Reranker weights resident): rerank candidates
  wait(rag_sem, 3), signal(rag_sem, 4)
  frontier: {cpu0_axis: ..., dgpu0_axis: ..., dgpu1_axis: ...}

GPU 0 (Llama weights resident): generate answer with context
  wait(rag_sem, 4), signal(rag_sem, 5)
  frontier: {cpu0_axis: ..., dgpu0_axis: ..., dgpu1_axis: ...}
```

The RAG pipeline spans both GPUs and the CPU. Each step is placed on the
device that already has the relevant weights resident, avoiding paging.
The frontier at the final step carries axes from all three compute domains,
documenting the complete provenance of the answer.

If the reranker's weights are not resident on GPU 1 (they were deallocated
to make room for SDXL), the scheduler can either:

- Load them from disk (adding the DMA axis to the chain), or
- Run the reranker on the CPU (different axis, same frontier propagation),
  whichever meets the latency target.

The frontier system treats these placement decisions identically — the
causal structure adapts to wherever the work actually runs. The
[visualizer's scenario 12](visualizer/?scenario=rag-pipeline)
shows work bouncing between CPU, GPU 0, and GPU 1 based on weight residency,
with the final frontier carrying axes from all three compute domains.

### Scenario B4: Streaming Translation Pipeline

Live voice-to-voice translation — a "babelfish" — chains six stages into a
streaming pipeline: VAD segments incoming audio, an encoder converts audio
features to a latent representation, a decoder produces source-language tokens,
an LLM translates to the target language, TTS synthesizes speech, and playback
delivers it.

```text
CPU: VAD (voice activity detection)
  signal(vad_sem, N)

GPU: Encoder (audio features → latent)
  wait(vad_sem, N), signal(encoder_sem, N)

GPU: Decoder / STT (latent → source tokens)
  wait(encoder_sem, N), signal(stt_sem, N)     ← chain-through point

GPU: LLM Translation (source → target tokens)
  wait(stt_sem, N), signal(llm_sem, N)         ← bottleneck (3 ticks)

GPU: TTS (target tokens → audio)
  wait(llm_sem, N), signal(tts_sem, N)

CPU: Playback
  wait(tts_sem, N), signal(playback_sem, N)
```

The pipeline chains on the STT output: the next speech chunk's VAD can start
as soon as the current chunk's decoder produces tokens. With a single GPU,
every GPU stage serializes and there is no inter-chunk overlap. With two GPUs,
the second GPU can encode chunk N+1 while the LLM translates chunk N — the
encoder and LLM occupy separate lanes with independent scheduling.

Two stages carry state across chunk boundaries, creating inter-iteration
dependencies that "skip" over the normal pipeline chain:

- **Decoder state**: The autoregressive decoder maintains beam search
  hypotheses and hidden state from chunk to chunk. Decode(N+1) depends
  directly on Decode(N)'s STT semaphore, not just transitively through the
  pipeline chain. This skip dependency ensures the frontier at Decode(N+1)
  carries Decode(N)'s full causal history — including everything upstream
  of the previous chunk's decoding.

- **TTS prosody state**: TTS must maintain pitch contour, speaking rate,
  and voice continuity across chunks. Synthesize(N+1) depends directly on
  Synthesize(N)'s TTS semaphore. Without this skip, Synthesize(N+1) would
  have no causal connection to Synthesize(N) at all — the normal pipeline
  chain flows through Decode → Translate, bypassing the previous
  Synthesize entirely. The skip ensures prosody state propagates.

These skip dependencies create cross-iteration arrows in the dependency graph
that bypass intermediate stages. The frontier at Synthesize(N+1) inherits
not just the current chunk's translation, but also the previous chunk's
complete TTS context — including, transitively, the previous chunk's
translation, decoding, and encoding. The scheduler can verify inter-iteration
state continuity by checking that the frontier at each skip target dominates
the previous iteration's skip source.

The LLM translation stage is the per-chunk latency bottleneck at 3 ticks (the
longest single stage). Pipeline throughput, however, is governed by the
chain-through interval: VAD + Encode + Decode = 5 ticks between successive
chunk starts. With 2 GPUs, both lanes stay busy — one GPU translates chunk N
while the other encodes chunk N+1 — yielding roughly one completed chunk
every 5 ticks in steady state. The frontier at each stage carries the complete
upstream provenance — the playback stage's frontier includes the VAD, encoder,
decoder, LLM, and TTS axes, documenting the full audio-to-audio causal chain.

In a real system, the pipeline stages would have much finer granularity —
the LLM emitting tokens one at a time, each feeding the TTS incrementally.
The frontier mechanics are identical: each token emission is a semaphore
signal, and the TTS can begin synthesizing as soon as the first token arrives.
The [visualizer's scenario 14](visualizer/?scenario=streaming-translation)
shows the coarse-grained version; use the depth slider (up to 8 chunks) with
2 GPUs to see the pipeline fill and reach steady state. The cross-iteration
skip arrows are visible in both the dependency graph and the timeline at
depth ≥ 2.

### What this demonstrates

The workstation scenarios show:

- **Cross-GPU pipelining** without P2P (host-mediated transfers participate
  in the frontier chain)
- **NVMe weight paging** as a first-class participant in the causal system
- **Dynamic placement** across GPUs based on weight residency
- **Gap-filling** with background work during pipeline stalls
- **Memory sharing** across models through death frontier-gated reuse
- **Streaming pipeline overlap** across speech chunks, with throughput
  limited by the bottleneck stage regardless of pipeline depth

---

## Hardware Configuration C: Datacenter (MI300X Cluster)

### Hardware (per node)

- 2 CPUs (96 cores each)
- 8 MI300X GPUs (192GB HBM3 each, 1.5TB total, connected via XGMI/Infinity Fabric)
- 8 NVMe SSDs (14GB/s read each)
- 1 RDMA NIC (400Gbps InfiniBand or RoCE)

### Models (12+ concurrent, spanning 2 nodes)

| # | Model | Type | Size | Placement | Notes |
|---|-------|------|------|-----------|-------|
| 1 | Llama-3-70B | LLM | 140GB (FP16) | 4-way TP on GPUs 0-3 | Primary chat |
| 2 | Llama-3-8B | LLM (draft) | 16GB | GPU 0 | Speculative decoding |
| 3 | Mixtral-8x7B | MoE LLM | ~90GB (all experts) | GPUs 4-5 | Dynamic expert selection |
| 4 | Code-Llama-34B | LLM | 68GB | 2-way TP on GPUs 6-7 | Code completion |
| 5 | Whisper-large-v3 | STT | 4GB | Any GPU with capacity | Voice input |
| 6 | XTTS-v2 | TTS | 4GB | Any GPU with capacity | Voice output |
| 7 | SDXL-Turbo | Diffusion | 12GB | Any GPU with capacity | Fast image gen |
| 8 | BGE-M3 | Embedding | 2GB | Any GPU | Multi-language embedding |
| 9 | Jina-ColBERT-v2 | Late-interaction | 2GB | Any GPU | Multi-vector retrieval |
| 10 | LayoutLMv3 | Document | 1.5GB | Any GPU | OCR + layout |
| 11 | Llama-3-70B (node 2) | LLM | 140GB | 4-way TP on remote GPUs 0-3 | Overflow / pipeline-parallel |
| 12 | Custom MoE (collective) | Cross-node MoE | ~200GB | GPUs on both nodes | Expert routing across nodes |

Total weight storage: ~680GB across two nodes (~1.5TB HBM3 per node = 3TB
total). Everything fits in HBM, but active KV caches and activations require
careful management. With 50+ concurrent chat sessions, KV cache pressure
drives NVMe paging.

### Scenario C1: Tensor-Parallel 70B with Gap-Filling

Llama-3-70B runs 4-way tensor parallel on GPUs 0-3. Each transformer layer
has an all-reduce synchronization point where all 4 GPUs exchange partial
results. During these sync windows, each GPU is partially idle — the compute
units that are not participating in the collective communication have nothing
to do.

**Without gap-filling**: The idle compute cycles are wasted. At 30
transformer layers with ~100us per all-reduce, that is ~3ms of idle time
per token generation step across 4 GPUs.

**With frontier-based gap-filling**:

```text
GPUs 0-3: Llama-70B attention layer N
  signal(tp_collective_sem, 2*N)  // one collective axis for 4 GPUs
  frontier: {tp_collective: 2*N}

GPUs 0-3: Llama-70B all-reduce (partial idle)
  During idle window on each GPU:
    GPU 0: BGE embedding batch (independent axis: {bge_axis: K})
    GPU 1: Whisper encoder chunk (independent axis: {whisper_axis: M})
    GPU 2: (idle, or more BGE)
    GPU 3: (idle, or LayoutLM inference)

GPUs 0-3: Llama-70B FFN layer N
  wait(tp_collective_sem, 2*N), signal(tp_collective_sem, 2*N+1)
```

The gap-filling work has disjoint frontier axes from the Llama-70B collective.
The scheduler can interleave it freely — dominance checking confirms
independence in O(k) time. The collective channel ensures Llama-70B's
synchronization appears as a single frontier entry regardless of the number
of participating GPUs.

The practical utilization gain depends on the workload mix and all-reduce
duration, but even filling 30-50% of collective sync gaps with embedding
or small-model inference can improve aggregate throughput by 10-15% — the
equivalent of running a "free" additional model during Llama-70B's sync
stalls.

### Scenario C2: Cross-Node Pipeline-Parallel with KV Cache Paging

Two nodes run Llama-3-70B with pipeline parallelism: the first 32 layers
on node 1's GPUs 0-3, the last 48 layers on node 2's GPUs 0-3. The RDMA
NIC transfers activations between nodes.

```text
Node 1 GPUs 0-3: layers 0-31 (TP collective within node)
  signal(pipeline_sem, S)
  frontier: {node1_tp_collective: ..., pipeline: S}

Node 1 NIC: RDMA send (activations to node 2)
  wait(pipeline_sem, S), signal(network_sem, N)
  frontier: {node1_tp_collective: ..., pipeline: S, nic1_axis: N}

Node 2 NIC: RDMA receive
  signal(network_sem, N)  // node 2's perspective
  frontier: {nic2_axis: N}  // inherits node 1's frontier via the data

Node 2 GPUs 0-3: layers 32-79 (TP collective within node)
  wait(network_sem, N), signal(pipeline_sem, S+1)
  frontier: {node1_tp_collective: ..., node2_tp_collective: ...,
             nic1_axis: N, nic2_axis: N, pipeline: S+1}
```

The pipeline semaphore spans both nodes. Node 2's GPUs can queue the next
pipeline stage before node 1 has started computing — the RDMA transfer
and subsequent compute are pipelined on the hardware FIFOs. When node 2
receives the activation data, its frontier carries node 1's collective
axis transitively, so node 2's scheduler knows the full dependency chain.

**KV cache paging under memory pressure**:

With 50 concurrent sessions, the total KV cache requirement exceeds
available HBM. The system pages cold KV cache blocks to NVMe:

```text
Page manager on node 1:
  Identify LRU KV blocks across all sessions
  NVMe DMA: evict blocks → signal(page_sem, P++)

When session resumes:
  NVMe DMA: load blocks → signal(page_sem, P++)
  GPU compute: wait(page_sem, P), continue decoding
  frontier includes {nvme_axis: P, ...}
```

The page manager operates globally across all sessions and models on the
node. It uses a single LRU ordered across all KV cache blocks, not per-model
LRU. A block from session 47 of Llama-70B competes for residency with a
block from session 3 of Code-Llama — the one accessed most recently stays.

The death frontier on each evicted block captures exactly which compute
has completed, so restoration knows precisely when the memory is safe to
reclaim. No per-session event tracking is needed — one death frontier per
block, regardless of how many decode steps read it.

### Scenario C3: Dynamic MoE Expert Routing

Mixtral-8x7B activates 2 of 8 experts per token. The active experts vary
per token. Experts are sharded across GPUs 4-5, with some experts resident
on NVMe for cold routing patterns.

```text
GPU 4 or 5: Mixtral router network
  Determines expert indices for this token
  signal(moe_sem, T)

For each selected expert:
  If expert weights resident on target GPU:
    GPU: expert FFN → signal(expert_sem, E)
    frontier: {moe_router: T, gpu_axis: ...}

  If expert weights on NVMe (cold expert):
    NVMe DMA: load expert weights → signal(page_sem, P)
    GPU: wait(page_sem, P), expert FFN → signal(expert_sem, E)
    frontier: {moe_router: T, nvme_axis: P, gpu_axis: ...}

GPU: combine expert outputs
  wait(expert_sem, E for each selected expert)
  signal(moe_sem, T+1)
  frontier: merge of all expert frontiers
```

The MoE routing decision is dynamic — different tokens activate different
experts. The frontier system handles this naturally: each expert computation
has its own frontier, and the combination step merges them. If one expert
required NVMe paging (cold expert), that delay appears in the frontier; if
both experts were resident, the combination step proceeds immediately.

For cross-node MoE (model #12), expert routing may select experts on the
other node. The activation data for those experts is sent via RDMA:

```text
Node 1: router selects expert on node 2
  NIC: RDMA send expert input → signal(network_sem, N)

Node 2: expert computation
  wait(network_sem, N), compute → signal(expert_result_sem, R)

Node 2 NIC: RDMA send result back → signal(network_sem, N+1)

Node 1: receive result, combine
  wait(network_sem, N+1)
  frontier: {moe_router: T, nic1: ..., nic2: ..., remote_gpu: ..., ...}
```

The cross-node expert routing adds NIC and remote GPU axes to the frontier.
Any downstream work waiting on this MoE output inherits the complete
dependency chain — including the network hops — through a single semaphore
wait.

### Scenario C4: Multi-Model Request Orchestration

A multimedia request arrives: the user uploads an image with a voice note
asking "what's in this picture and write code to generate something similar."

```text
Phase 1 (parallel preprocessing):
  CPU: audio decode → signal(audio_sem, 1)
  CPU: image decode → signal(image_sem, 1)

Phase 2 (parallel understanding):
  GPU (any): Whisper STT on audio
    wait(audio_sem, 1), signal(text_sem, 1)

  GPU (any): CLIP-ViT-L on image
    wait(image_sem, 1), signal(vision_sem, 1)

Phase 3 (multi-modal LLM):
  GPUs 0-3: Llama-70B with both text + vision context
    wait(text_sem, 1), wait(vision_sem, 1)
    signal(llama_sem, T)
    frontier: {cpu: ..., whisper_gpu: ..., clip_gpu: ..., tp_collective: T}

Phase 4 (code generation):
  GPUs 6-7: Code-Llama-34B generates code based on Llama's description
    wait(llama_sem, T), signal(code_sem, C)
    frontier: includes all upstream axes transitively

Phase 5 (image generation):
  GPU (any): SDXL-Turbo generates image from Code-Llama's code description
    wait(code_sem, C), signal(sdxl_sem, 1)
    frontier: captures entire pipeline provenance

Phase 6 (response):
  CPU: assemble text + code + image response
    wait(code_sem, C), wait(sdxl_sem, 1)
  GPU (any): XTTS synthesizes voice response
    wait(llama_sem, T), signal(tts_sem, 1)
```

This request touches 7 different models across multiple GPUs, the CPU,
and potentially the NIC (if CLIP or SDXL run on a different node). The
entire pipeline is submitted atomically at request arrival. Each model's
work is placed on whatever GPU has the relevant weights resident and
available capacity.

The final response's frontier captures the complete multi-model provenance:
audio processing, image understanding, language modeling, code generation,
image synthesis, and voice synthesis — all expressed through the frontier
merge algebra, with no application-level dependency graph management. The
[visualizer's scenario 13](visualizer/?scenario=multi-model)
shows this pipeline; use the GPU slider to see how two GPUs allow the
understanding phase (Whisper + CLIP) and output phase (Code-Llama + TTS) to
parallelize.

### What this demonstrates

The datacenter scenarios show:

- **Collective channels** compressing 4-GPU or 8-GPU TP synchronization
  to a single frontier entry
- **Gap-filling** during collective sync stalls with independent model work
- **Cross-node pipeline parallelism** with RDMA transfers as first-class
  frontier participants
- **Global KV cache paging** with LRU across all models and sessions
- **Dynamic MoE routing** with variable expert placement (GPU or NVMe)
  expressed through the same frontier system
- **Multi-model request orchestration** with 7+ models in a single
  atomically-submitted pipeline
- **NVMe, NIC, CPU, and GPU** all participating as frontier axes in a
  unified scheduling substrate

---

## Common Patterns Across All Configurations

Several patterns recur regardless of hardware scale:

**Atomic pipeline submission**: The client submits the entire multi-stage
pipeline before any work begins. Hardware FIFOs and semaphore ordering
ensure correct execution. This eliminates round-trips between stages and
allows the scheduler to see the complete work graph for placement decisions.

**Death frontier-based memory sharing**: All models share a common buffer
pool. Buffers freed by one model can be reused by another as soon as the
frontier dominance check passes. No per-model memory partitioning, no IPC
for cross-model buffer sharing.

**Disjoint axis isolation**: Independent models have non-overlapping frontier
axes. The scheduler can interleave their work on shared hardware with zero
causal interference. This is not a guarantee that must be maintained through
careful engineering — it falls out of the frontier algebra automatically.

**NVMe paging as a normal queue axis**: KV cache paging, checkpoint writes,
and weight reloads all participate in the same frontier system as compute and
network operations. The page manager does not need its own synchronization
substrate.

**Priority scheduling via operation ordering**: Higher-priority work is
submitted earlier in the queue. Lower-priority work fills gaps. The frontier
system provides the safety guarantees; the scheduler provides the policy.
