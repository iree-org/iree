// Scenario scaling and hardware expansion.
//
// Three orthogonal scaling axes:
//   - Width: N independent copies sharing hardware (concurrent requests).
//     Each width instance has independent semaphores.
//   - Depth: M sequential copies chained via chain_through and
//     iteration_deps semaphores. chain_through gates iteration start
//     (root ops wait), iteration_deps carry state between specific
//     operations across iterations (e.g., decoder hidden state).
//   - Hardware count: number of lanes per scalable hardware type.
//     Operations are dynamically assigned to lanes at simulation time.
//
// Combined, these show how non-FIFO scheduling packs multiple request
// graphs onto varying hardware configurations, and how frontiers
// accumulate through pipeline iterations.

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Scale a base scenario to width × depth instances.
// Returns the base scenario unmodified (reference-equal) when both are 1.
export function scaleScenario(base, width, depth) {
  if (width === 1 && depth === 1) return base;

  const chain_set = new Set(base.chain_through || []);

  // Build iteration_deps map: op_id -> Set<sem_id>.
  // These semaphores carry per-operation state between depth iterations.
  // Unlike chain_through (which gates root ops), iteration_deps inject
  // waits on specific operations — creating "skip" dependencies that
  // connect the same operation across iterations.
  const iteration_deps = new Map();
  if (base.iteration_deps) {
    for (const [op_id, sem_ids] of Object.entries(base.iteration_deps)) {
      iteration_deps.set(op_id, new Set(sem_ids));
    }
  }

  // Depth-shared semaphores: union of chain_through and iteration_deps.
  // These get one instance per width lane (shared across all depths) with
  // value offsets per depth, rather than one instance per (width, depth).
  const depth_shared_set = new Set(chain_set);
  for (const sem_ids of iteration_deps.values()) {
    for (const sem_id of sem_ids) {
      depth_shared_set.add(sem_id);
    }
  }

  // Maximum signal value per semaphore — used as the stride for depth offsets.
  const max_signal = {};
  for (const sem of base.semaphores) {
    max_signal[sem.id] = 0;
  }
  for (const op of base.operations) {
    for (const [sem_id, value] of Object.entries(op.signal)) {
      max_signal[sem_id] = Math.max(max_signal[sem_id], value);
    }
  }

  // Root operations: those with no waits in the base scenario.
  // At depth > 0, these get injected chain_through waits.
  const root_op_ids =
      new Set(base.operations.filter(op => Object.keys(op.wait).length === 0)
                  .map(op => op.id));

  // Hardware is shared across all instances — that's the whole point.
  const hardware = base.hardware;

  // --- Semaphores ---
  // Depth-shared: one per width lane, shared across depths.
  // Regular: one per (width, depth) instance.
  const semaphores = [];
  for (let w = 0; w < width; w++) {
    for (const sem of base.semaphores) {
      if (depth_shared_set.has(sem.id)) {
        semaphores.push({
          id: make_sem_id(sem.id, w, null, width, depth),
          label: make_sem_label(sem.label, w, null, width, depth),
        });
      }
    }
    for (let d = 0; d < depth; d++) {
      for (const sem of base.semaphores) {
        if (!depth_shared_set.has(sem.id)) {
          semaphores.push({
            id: make_sem_id(sem.id, w, d, width, depth),
            label: make_sem_label(sem.label, w, d, width, depth),
          });
        }
      }
    }
  }

  // --- Operations ---
  // Ordered width-first, then depth-within-width.
  // Definition order controls scheduling priority: earlier width instances
  // get first access to shared hardware, later ones fill idle slots.
  const operations = [];
  for (let w = 0; w < width; w++) {
    for (let d = 0; d < depth; d++) {
      for (const op of base.operations) {
        // Remap wait entries.
        const wait = {};
        for (const [sem_id, value] of Object.entries(op.wait)) {
          const is_shared = depth_shared_set.has(sem_id);
          const new_id =
              make_sem_id(sem_id, w, is_shared ? null : d, width, depth);
          const offset = is_shared ? d * max_signal[sem_id] : 0;
          wait[new_id] = value + offset;
        }

        // Root ops at depth > 0: inject waits on chain_through sems
        // at the previous depth's exit values.
        if (d > 0 && root_op_ids.has(op.id)) {
          for (const chain_sem of chain_set) {
            const new_id = make_sem_id(chain_sem, w, null, width, depth);
            wait[new_id] = d * max_signal[chain_sem];
          }
        }

        // iteration_deps at depth > 0: inject per-operation waits on the
        // previous depth's signal values. These create "skip" dependencies
        // that connect the same operation across iterations, carrying
        // inter-iteration state (e.g., decoder hidden state, TTS prosody).
        if (d > 0 && iteration_deps.has(op.id)) {
          for (const dep_sem of iteration_deps.get(op.id)) {
            const new_id = make_sem_id(dep_sem, w, null, width, depth);
            const dep_value = d * max_signal[dep_sem];
            // Use max in case the operation also has a normal wait on the
            // same depth-shared semaphore (the stronger constraint wins).
            wait[new_id] = Math.max(wait[new_id] || 0, dep_value);
          }
        }

        // Remap signal entries.
        const signal = {};
        for (const [sem_id, value] of Object.entries(op.signal)) {
          const is_shared = depth_shared_set.has(sem_id);
          const new_id =
              make_sem_id(sem_id, w, is_shared ? null : d, width, depth);
          const offset = is_shared ? d * max_signal[sem_id] : 0;
          signal[new_id] = value + offset;
        }

        operations.push({
          id: make_op_id(op.id, w, d, width, depth),
          hardware: op.hardware,
          label: make_op_label(op.label, w, d, width, depth),
          duration: op.duration,
          wait,
          signal,
        });
      }
    }
  }

  return {
    id: `${base.id}_${width}x${depth}`,
    name: `${base.name} (${width}\u00d7${depth})`,
    description: base.description,
    hardware,
    semaphores,
    operations,
    annotations: [],
  };
}

// Check if a scenario supports depth chaining.
export function supportsDepth(scenario) {
  const has_chain = Array.isArray(scenario.chain_through) &&
      scenario.chain_through.length > 0;
  const has_iteration_deps = scenario.iteration_deps &&
      Object.keys(scenario.iteration_deps).length > 0;
  return has_chain || has_iteration_deps;
}

// Find the scalable hardware entry in a scenario, if any.
// Scalable hardware has a `count` field indicating the default lane count.
export function getScalableHardware(scenario) {
  return scenario.hardware.find(hw => hw.count !== undefined) || null;
}

// Expand count-based hardware into individual lanes for simulation.
//
// Hardware entries with `count` are expanded into N individual lane entries
// with `class` and `class_label` fields. Operations keep their original
// hardware reference (the class name), and the simulator assigns them to
// specific lanes at issue time.
//
// When count <= 1, the scenario is returned unchanged — the single lane's
// ID matches the class name, so operations' hardware references work as
// direct lane references without expansion.
export function expandHardware(scenario, count_override) {
  const scalable = getScalableHardware(scenario);
  if (!scalable) return scenario;

  const count = count_override ?? scalable.count;
  if (count <= 1) return scenario;

  const new_hardware = [];
  for (const hw of scenario.hardware) {
    if (hw.id === scalable.id) {
      for (let i = 0; i < count; i++) {
        new_hardware.push({
          id: `${hw.id}_${i}`,
          label: `${hw.label} ${i}`,
          class: hw.id,
          class_label: hw.label,
        });
      }
    } else {
      new_hardware.push(hw);
    }
  }

  return {
    ...scenario,
    hardware: new_hardware,
  };
}

// ---------------------------------------------------------------------------
// Naming helpers
// ---------------------------------------------------------------------------

function make_sem_id(base_id, w, d, width, depth) {
  // d === null means depth-shared (chain_through or iteration_deps).
  let id = base_id;
  if (width > 1) id += `_w${w}`;
  if (d !== null && depth > 1) id += `_d${d}`;
  return id;
}

function make_sem_label(base_label, w, d, width, depth) {
  if (width === 1 && d === null) return base_label;
  if (width > 1 && (d === null || depth === 1)) return `${base_label} [${w}]`;
  if (width === 1 && depth > 1) return `${base_label} [d${d}]`;
  return `${base_label} [${w},${d}]`;
}

function make_op_id(base_id, w, d, width, depth) {
  return `${base_id}_w${w}_d${d}`;
}

function make_op_label(base_label, w, d, width, depth) {
  if (width > 1 && depth > 1) return `${base_label} [${w},${d}]`;
  if (width > 1) return `${base_label} [${w}]`;
  if (depth > 1) return `${base_label} [d${d}]`;
  return base_label;
}
