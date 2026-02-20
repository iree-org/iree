// Vector Clock & Frontier Simulator
//
// Simulates the execution of a scenario, producing state snapshots at each
// logical tick. The core model:
//
//   - Semaphores are the clock axes. Each semaphore is a monotonically
//     increasing timeline. Operations signal semaphores to advance their
//     value and wait on semaphores to establish ordering.
//
//   - Hardware lanes are where operations physically execute. Each lane
//     can run one operation at a time. Hardware sharing is a scheduling
//     constraint, not a causal relationship.
//
//   - Frontiers are vectors over semaphore space: {sem_id: value, ...}.
//     An operation's frontier is computed by merging the frontiers from
//     all its semaphore waits plus its own signal values. Hardware lane
//     assignment does NOT affect frontier propagation.
//
//   - The >= property: waiting on sem >= K is satisfied whenever the
//     semaphore's current value is at least K. The waiter inherits the
//     frontier from the signal that set the semaphore to value K — NOT
//     the semaphore's current frontier. Scheduling is decoupled from
//     causality: the wait resolves immediately, but the frontier only
//     captures the causal dependency on the specific signal waited on.

// ---------------------------------------------------------------------------
// Frontier operations
// ---------------------------------------------------------------------------

// Merge multiple frontiers by taking the component-wise maximum.
// This is the "join" operation for vector clocks:
//   merge({a: 3, b: 1}, {a: 1, c: 5}) = {a: 3, b: 1, c: 5}
export function mergeFrontiers(...frontiers) {
  const result = {};
  for (const frontier of frontiers) {
    for (const [sem, value] of Object.entries(frontier)) {
      result[sem] = Math.max(result[sem] || 0, value);
    }
  }
  return result;
}

// Check if frontier `a` dominates frontier `b`.
// a dominates b iff for every semaphore in b, a[sem] >= b[sem].
// This represents the "happens-before" relation: if a dominates b, then
// everything b depends on has already happened according to a's knowledge.
export function dominates(a, b) {
  for (const [sem, value] of Object.entries(b)) {
    if ((a[sem] || 0) < value) return false;
  }
  return true;
}

// Format a frontier as a human-readable string for display.
export function frontierToString(frontier) {
  const entries = Object.entries(frontier);
  if (entries.length === 0) return '\u2205';
  return '{ ' + entries.map(([k, v]) => `${k}:${v}`).join(', ') + ' }';
}

function cloneFrontier(frontier) {
  return {...frontier};
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

// Run the discrete-event simulation for a scenario.
// Returns an array of snapshots, one per tick, capturing the complete state
// of all operations and semaphores at each moment.
export function simulate(scenario) {
  const snapshots = [];

  // Per-operation mutable state.
  const op_states = {};
  for (const op of scenario.operations) {
    op_states[op.id] = {
      state: 'pending',
      start_tick: null,
      end_tick: null,
      frontier: {},
      assigned_lane: null,
    };
  }

  // Per-semaphore mutable state.
  const sem_states = {};
  for (const sem of scenario.semaphores) {
    sem_states[sem.id] = {
      value: 0,
      frontier: {},   // frontier of the latest signal (for display)
      frontiers: {},  // per-value frontier history (for causal lookup)
    };
  }

  // Build class-to-lanes mapping for scalable hardware.
  // Hardware entries with a `class` field were expanded from count-based
  // hardware. Operations referencing the class name get dynamically assigned
  // to available lanes at issue time.
  const class_to_lanes = {};
  for (const hw of scenario.hardware) {
    if (hw.class) {
      if (!class_to_lanes[hw.class]) class_to_lanes[hw.class] = [];
      class_to_lanes[hw.class].push(hw.id);
    }
  }

  // Group operations by hardware type (class name or direct lane ID).
  const ops_by_type = {};
  for (const op of scenario.operations) {
    if (!ops_by_type[op.hardware]) ops_by_type[op.hardware] = [];
    ops_by_type[op.hardware].push(op);
  }

  let tick = 0;
  const MAX_TICKS = 500;

  while (tick <= MAX_TICKS) {
    const events = [];

    // Within a single tick, state transitions can cascade: a completion can
    // satisfy a wait, enabling an immediate issue. We loop until no more
    // transitions occur (fixpoint within the tick).
    let changed = true;
    while (changed) {
      changed = false;

      // Phase 1: Complete operations whose duration has elapsed.
      for (const op of scenario.operations) {
        const os = op_states[op.id];
        if (os.state !== 'in_flight') continue;
        if (tick < os.start_tick + op.duration) continue;

        os.state = 'retired';
        os.end_tick = tick;
        changed = true;

        // Signal semaphores with this operation's frontier.
        for (const [sem_id, value] of Object.entries(op.signal)) {
          if (sem_states[sem_id].value < value) {
            sem_states[sem_id].value = value;
            sem_states[sem_id].frontier = cloneFrontier(os.frontier);
            sem_states[sem_id].frontiers[value] = cloneFrontier(os.frontier);
            events.push({
              type: 'signaled',
              op_id: op.id,
              description: `${sem_label(scenario, sem_id)} \u2192 ${value}`,
            });
          }
        }

        events.push({
          type: 'retired',
          op_id: op.id,
          description: `${op.label} retired`,
        });
      }

      // Phase 2: Check if pending operations become ready.
      // An operation is ready when ALL its semaphore waits are satisfied.
      // No FIFO ordering — semaphores are the only ordering mechanism.
      for (const op of scenario.operations) {
        const os = op_states[op.id];
        if (os.state !== 'pending') continue;

        let waits_satisfied = true;
        for (const [sem_id, value] of Object.entries(op.wait)) {
          if ((sem_states[sem_id]?.value || 0) < value) {
            waits_satisfied = false;
            break;
          }
        }
        if (!waits_satisfied) continue;

        os.state = 'ready';
        changed = true;
        events.push({
          type: 'ready',
          op_id: op.id,
          description: `${op.label} ready`,
        });
      }

      // Phase 3: Issue ready operations to idle hardware lanes.
      // For each hardware type (class or direct lane), find free lanes and
      // assign ready operations in definition order. When a hardware class
      // has multiple lanes (e.g., 3 GPUs), multiple operations can issue
      // simultaneously. This is a scheduling decision, NOT a causal
      // relationship — no frontier propagation from hardware assignment.

      // Build set of occupied lanes for this iteration.
      const occupied_lanes = new Set();
      for (const op of scenario.operations) {
        if (op_states[op.id].state === 'in_flight') {
          occupied_lanes.add(op_states[op.id].assigned_lane);
        }
      }

      for (const [hw_type, type_ops] of Object.entries(ops_by_type)) {
        // Resolve lanes: class-based hardware maps to expanded lanes,
        // direct-lane hardware maps to itself.
        const lanes = class_to_lanes[hw_type] || [hw_type];
        const free_lanes = lanes.filter(lane => !occupied_lanes.has(lane));
        if (free_lanes.length === 0) continue;

        const ready_ops =
            type_ops.filter(op => op_states[op.id].state === 'ready');
        if (ready_ops.length === 0) continue;

        // Assign operations to lanes: definition order for ops (maintained
        // by array order), first available for lanes.
        const assign_count = Math.min(free_lanes.length, ready_ops.length);
        for (let i = 0; i < assign_count; i++) {
          const ready_op = ready_ops[i];
          const os = op_states[ready_op.id];
          os.state = 'in_flight';
          os.start_tick = tick;
          os.assigned_lane = free_lanes[i];
          occupied_lanes.add(free_lanes[i]);
          changed = true;

          // Build the operation's frontier by merging:
          //   1. The frontier from each waited semaphore at the specific
          //      value waited on (NOT the semaphore's current frontier —
          //      causality tracks the specific signal, not later advances)
          //   2. This operation's own signal values (its contribution)
          //
          // The hardware lane does NOT contribute to the frontier. Two
          // independent operations on the same GPU share no causal info.
          const wait_frontiers =
              Object.entries(ready_op.wait)
                  .filter(([sem_id]) => sem_states[sem_id])
                  .map(
                      ([sem_id, wait_value]) =>
                          sem_states[sem_id].frontiers[wait_value] || {});

          os.frontier = mergeFrontiers(
              ...wait_frontiers,
              ready_op.signal,
          );

          events.push({
            type: 'issued',
            op_id: ready_op.id,
            description: `${ready_op.label} issued ` +
                `(frontier: ${frontierToString(os.frontier)})`,
          });
        }
      }
    }

    // Record a deep-cloned snapshot of the full state at this tick.
    snapshots.push({
      tick,
      operations: Object.fromEntries(scenario.operations.map(
          op =>
              [op.id,
               {
                 state: op_states[op.id].state,
                 start_tick: op_states[op.id].start_tick,
                 end_tick: op_states[op.id].end_tick,
                 frontier: cloneFrontier(op_states[op.id].frontier),
                 assigned_lane: op_states[op.id].assigned_lane,
               },
    ])),
      semaphores: Object.fromEntries(scenario.semaphores.map(
          sem =>
              [sem.id,
               {
                 value: sem_states[sem.id].value,
                 frontier: cloneFrontier(sem_states[sem.id].frontier),
               },
    ])),
      events: [...events],
    });

    // All operations retired — add one final empty-event tick and stop.
    const all_retired =
        scenario.operations.every(op => op_states[op.id].state === 'retired');
    if (all_retired) {
      snapshots.push({
        tick: tick + 1,
        operations: snapshots[snapshots.length - 1].operations,
        semaphores: snapshots[snapshots.length - 1].semaphores,
        events: [],
      });
      break;
    }

    tick++;
  }

  return snapshots;
}

// ---------------------------------------------------------------------------
// Dependency computation
// ---------------------------------------------------------------------------

// Extract dependency edges from a scenario: for each (semaphore, value) pair
// that one operation signals and another waits on, create an edge. These
// edges define the DAG structure shown in the work graph view.
export function computeDependencies(scenario) {
  const deps = [];
  for (const waiter of scenario.operations) {
    for (const [sem_id, wait_value] of Object.entries(waiter.wait)) {
      const signaler =
          scenario.operations.find(op => op.signal[sem_id] === wait_value);
      if (signaler) {
        deps.push({
          from: signaler.id,
          to: waiter.id,
          semaphore: sem_id,
          value: wait_value,
        });
      }
    }
  }
  return deps;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sem_label(scenario, sem_id) {
  const sem = scenario.semaphores.find(s => s.id === sem_id);
  return sem ? sem.label : sem_id;
}
