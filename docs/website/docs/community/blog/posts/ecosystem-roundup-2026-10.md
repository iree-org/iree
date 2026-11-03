---
date: 2026-11-03
authors:
  - rodburns
categories:
  - Community
tags:
  - Ecosystem
  - Releases
---

# IREE ecosystem roundup: October 2026

![IREE newsletter, October 2026](./ecosystem-roundup-2026-10-banner.png)

Welcome to the October roundup of news from across the IREE ecosystem.

Two IREE-based hardware launches led October: Google released Coral NPU
compiler v1.0, and Synaptics launched its Astra SR200 processors, which use the
IREE compiler and runtime. Maximilian Bartel also became chair of the IREE
Technical Steering Committee (TSC).

Below, you'll find these stories alongside LLVM Developers' Meeting and PyTorch
Conference talks, the IREE 3.12 release, and highlights from the community.

## Google and Synaptics ship edge AI hardware built on IREE

On 21 October, Google released
[Coral NPU compiler v1.0](https://developers.googleblog.com/unifying-edge-ai-how-coral-npus-open-standard-brings-portable-energy-viable-intelligence-from-model-to-silicon/),
which is based on IREE, and Synaptics launched the Astra SR200 series on top of
Coral NPU, describing the chips as a new class of multimodal edge AI
microcontrollers that also use the IREE compiler and runtime. Together, the two
launches put IREE underneath commercial edge AI silicon from Synaptics and
Google Research.

You can
[buy the Coralboard SL development board at DigiKey](https://www.digikey.com/en/products/detail/synaptics-coralboard/720-KDVSL2619-11/29784108);
SR200 chips are still sampling.

Synaptics partner Enerzai has also published hands-on work with the IREE-based
Torq compiler:
[optimizing the Torq NPU compiler](https://medium.com/@enerzai/optimizing-the-mlir-based-torq-npu-compiler-f5ea6628c95d),
[porting a ternary speech-to-text model to the Torq NPU](https://medium.com/@enerzai/porting-a-ternary-stt-model-to-the-synaptics-torq-npu-bf9f55a6910e),
and building a
[low-memory smart-home agent demo](https://www.youtube.com/watch?v=YZ5QPGJJpDA).

## Maximilian Bartel now chairs the IREE Technical Steering Committee

Maximilian Bartel, Roofline's CTO, is now Chair of the IREE TSC. In the role,
he coordinates the TSC's work and presides over its meetings, while technical
oversight for IREE stays with the TSC as a whole. Read the
[TSC chair announcement](https://www.roofline.ai/news/news-maximilian-bartel-iree-tsc-chair)
on the Roofline blog. (The Edge AI Foundation also
[shared the news on LinkedIn](https://lnkd.in/p/gVF-_wep).)

## MIPS, GlobalFoundries, and Qualcomm back IREE in LLVM talks

At the [LLVM Developers' Meeting](https://llvm.swoogo.com/2026devmtg), Roofline
engineers presented joint work: RISC-V work with MIPS by GlobalFoundries, and
Hexagon NPU work with Qualcomm.
[You can see an overview of the LLVM Developers' Meeting speakers](https://www.roofline.ai/news/news-roofline-llvm-developers-meeting-2026)
on the Roofline website, and the presentation recordings will be shared once
they're available.

## Toradex, hiPEAC, and PyTorch Conference

* Roofline is now listed in the
  [Toradex partner network](https://www.toradex.com/support/partner-network/services/100081/roofline),
  enabling AI workloads on NXP i.MX 95 hardware using IREE. Toradex
  [announced the partnership on LinkedIn](https://lnkd.in/p/gETQRgGP).
* hiPEAC magazine published
  [an article written by Roofline CEO Dr. Jan Moritz Joseph](https://www.hipeac.net/files/public/33/7175/hipeacinfo79_web-8HHhWYGG.pdf)
  about how the team brought the latest mixture of experts (MoE) models to
  hardware using IREE.
* Rod Burns presented the IREE project at PyTorch Conference and met attendees
  at the
  [LF AI & Data booth](https://www.linkedin.com/feed/update/urn:li:activity:7518696528446922753).
  [Read the slides and recap](https://lnkd.in/p/g-T_V2b5).

## IREE 3.12 brings a rewritten Vulkan driver

[IREE 3.12](https://github.com/iree-org/iree/releases/tag/v3.12.0) shipped with
a rewritten Vulkan driver and an experimental hardware abstraction layer (HAL)
for AMD's Heterogeneous System Architecture (HSA), alongside an updated
[PyPI package](https://pypi.org/project/iree-base-compiler/). Read
[Phoronix's coverage of the release](https://phoronix.com/news/IREE-3.12-Released).

## IREE community meeting

Tobias Fuchs talked about the new features in the IREE 3.12 release and
outlined how the release cycle will work in the future. There was also a
presentation from Sebastian Fischer from Ludwig-Maximilians-University in
Munich on their project "anvl", which implements the R language and is
investigating the use of IREE to compile for different targets. Watch the
recording [on YouTube](https://www.youtube.com/watch?v=Y82MWrbhe54).

## The IREE community built backends, products, and tutorials

Community work this month ranged from a new POWER backend to automotive and
chip-design products. A few highlights:

* **IBM Research:** [library-free ML inference using IREE on POWER](https://drive.google.com/file/d/1q30e2pP7IULbbbbT7g6VPaYR7YPC8dZm/view)
* **Embecosm:** [bringing up IREE on a bare-metal microcontroller](https://embecosm.com/2026/09/22/howto-bringing-up-iree-up-on-a-bare-metal-microcontroller-full-series/)
* **Silicore AI:** [developing an IREE-based stack for automotive](https://lnkd.in/p/giuVuxjM)
* **Zenedgex:** [AI-powered chip-design exploration tools that use IREE](https://lnkd.in/p/gvksZ5MU)
* **Tobias Grosser, University of Cambridge:** [Quidditch, an end-to-end deep
  learning compiler for Occamy using IREE and xDSL](https://www.youtube.com/watch?v=KEMSeN-rGJY)
* **Waiz Khan:** [benchmarking dynamic mixture-of-experts (MoE) execution with
  IREE](https://www.linkedin.com/pulse/benchmarking-dynamic-moe-execution-iree-when-faster-result-waiz-khan-guusf/)
* **Jonathan Song, Ghent University:** [building an LLM compiler for an AI
  accelerator](https://lnkd.in/p/gAqGt-Aw), [plus a follow-up post](https://lnkd.in/p/ggWmhFSn)
* **Alex Denisov:** [a beginner's guide to MLIR's linalg dialect](https://lowlevelbits.com/p/beginners-guide-to-mlirs-linalg-dialect)

<!-- TODO: link each community highlight -->

Building something with IREE? We'd like to hear about it. Share it on the
[IREE Discord server](https://discord.gg/wEWh6Z9nMU) or the
[iree-technical-discussion email list](https://lists.lfaidata.foundation/g/iree-technical-discussion).
