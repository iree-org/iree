#!/usr/bin/env node
// validate.mjs — Structural and simulation validation for all scenarios.
//
// Validates base scenario integrity, chain_through fields, hardware expansion,
// and runs scaled simulations at multiple width × depth × hardware count
// levels to verify termination and frontier correctness.
//
// Usage: node validate.mjs

import {scenarios} from './scenarios.js';
import {simulate, frontierToString} from './simulator.js';
import {scaleScenario, supportsDepth, getScalableHardware, expandHardware,} from './scaling.js';

let errors = 0;
let warnings = 0;

function error(scenario, message) {
  console.error(`  ERROR: ${message}`);
  errors++;
}

function warn(scenario, message) {
  console.warn(`  WARN: ${message}`);
  warnings++;
}

function ok(message) {
  console.log(`  ${message}`);
}

// ---------------------------------------------------------------------------
// Base scenario structural checks
// ---------------------------------------------------------------------------

function check_structure(scenario) {
  const op_ids = scenario.operations.map(o => o.id);
  const sem_ids = scenario.semaphores.map(s => s.id);
  const hw_ids = scenario.hardware.map(h => h.id);

  // Unique IDs.
  if (new Set(op_ids).size !== op_ids.length) {
    error(scenario, 'duplicate operation IDs');
  }
  if (new Set(sem_ids).size !== sem_ids.length) {
    error(scenario, 'duplicate semaphore IDs');
  }
  if (new Set(hw_ids).size !== hw_ids.length) {
    error(scenario, 'duplicate hardware IDs');
  }

  const sem_set = new Set(sem_ids);
  const hw_set = new Set(hw_ids);

  for (const op of scenario.operations) {
    // Hardware reference valid.
    if (!hw_set.has(op.hardware)) {
      error(
          scenario,
          `op "${op.id}" references unknown hardware "${op.hardware}"`);
    }

    // Wait semaphores exist.
    for (const sem_id of Object.keys(op.wait)) {
      if (!sem_set.has(sem_id)) {
        error(scenario, `op "${op.id}" waits on unknown semaphore "${sem_id}"`);
      }
    }

    // Signal semaphores exist.
    for (const sem_id of Object.keys(op.signal)) {
      if (!sem_set.has(sem_id)) {
        error(scenario, `op "${op.id}" signals unknown semaphore "${sem_id}"`);
      }
    }

    // Signal values positive.
    for (const [sem_id, value] of Object.entries(op.signal)) {
      if (value <= 0 || !Number.isInteger(value)) {
        error(
            scenario,
            `op "${op.id}" signals "${sem_id}" with non-positive value ${
                value}`);
      }
    }
  }

  // No duplicate (semaphore, value) signals.
  const signal_pairs = new Map();
  for (const op of scenario.operations) {
    for (const [sem_id, value] of Object.entries(op.signal)) {
      const key = `${sem_id}:${value}`;
      if (signal_pairs.has(key)) {
        error(
            scenario,
            `duplicate signal ${key} in ops "${signal_pairs.get(key)}" and "${
                op.id}"`);
      }
      signal_pairs.set(key, op.id);
    }
  }

  // Every waited (sem, value) has a matching signal.
  for (const op of scenario.operations) {
    for (const [sem_id, value] of Object.entries(op.wait)) {
      const key = `${sem_id}:${value}`;
      if (!signal_pairs.has(key)) {
        error(
            scenario,
            `op "${op.id}" waits on ${key} but no operation signals it`);
      }
    }
  }

  // chain_through validation.
  if (scenario.chain_through) {
    for (const chain_sem of scenario.chain_through) {
      if (!sem_set.has(chain_sem)) {
        error(
            scenario,
            `chain_through references unknown semaphore "${chain_sem}"`);
      }
    }
  }

  // iteration_deps validation.
  const op_set = new Set(op_ids);
  if (scenario.iteration_deps) {
    for (const [dep_op, dep_sems] of Object.entries(scenario.iteration_deps)) {
      if (!op_set.has(dep_op)) {
        error(
            scenario,
            `iteration_deps references unknown operation "${dep_op}"`);
      }
      for (const dep_sem of dep_sems) {
        if (!sem_set.has(dep_sem)) {
          error(
              scenario,
              `iteration_deps["${dep_op}"] references unknown semaphore "${
                  dep_sem}"`);
        }
        // The operation should signal the referenced semaphore — otherwise the
        // inter-iteration dependency doesn't correspond to state carried by
        // that operation.
        const op_def = scenario.operations.find(o => o.id === dep_op);
        if (op_def && !(dep_sem in op_def.signal)) {
          error(
              scenario,
              `iteration_deps["${dep_op}"] references semaphore "${
                  dep_sem}" but the operation does not signal it`);
        }
      }
    }
  }

  // Annotation ticks within simulation range (checked after simulation).
}

// ---------------------------------------------------------------------------
// Simulation checks
// ---------------------------------------------------------------------------

// Simulate a scenario, applying hardware expansion if needed.
function prepare_and_simulate(scenario) {
  const expanded = expandHardware(scenario);
  return {snapshots: simulate(expanded), expanded};
}

function check_simulation(label, scenario) {
  const {snapshots, expanded} = prepare_and_simulate(scenario);
  const max_tick = snapshots.length - 1;

  if (max_tick >= 499) {
    error(scenario, `${label}: simulation did not terminate (hit MAX_TICKS)`);
    return null;
  }

  const final = snapshots[max_tick];
  const all_retired = expanded.operations.every(
      op => final.operations[op.id]?.state === 'retired');

  if (!all_retired) {
    const stuck =
        expanded.operations
            .filter(op => final.operations[op.id]?.state !== 'retired')
            .map(op => `${op.id}(${final.operations[op.id]?.state})`);
    error(scenario, `${label}: not all ops retired: ${stuck.join(', ')}`);
    return null;
  }

  return {snapshots, max_tick, op_count: expanded.operations.length};
}

// ---------------------------------------------------------------------------
// Scaled scenario structural checks
// ---------------------------------------------------------------------------

function check_scaled_structure(label, scenario) {
  const op_ids = scenario.operations.map(o => o.id);
  const sem_ids = scenario.semaphores.map(s => s.id);

  if (new Set(op_ids).size !== op_ids.length) {
    error(scenario, `${label}: duplicate operation IDs in scaled output`);
    return false;
  }
  if (new Set(sem_ids).size !== sem_ids.length) {
    error(scenario, `${label}: duplicate semaphore IDs in scaled output`);
    return false;
  }

  // Every waited (sem, value) has a matching signal.
  const signal_pairs = new Set();
  for (const op of scenario.operations) {
    for (const [sem_id, value] of Object.entries(op.signal)) {
      signal_pairs.add(`${sem_id}:${value}`);
    }
  }
  for (const op of scenario.operations) {
    for (const [sem_id, value] of Object.entries(op.wait)) {
      if (!signal_pairs.has(`${sem_id}:${value}`)) {
        error(
            scenario,
            `${label}: op "${op.id}" waits on ${sem_id}:${
                value} but no op signals it`);
        return false;
      }
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// Run all checks
// ---------------------------------------------------------------------------

for (const scenario of scenarios) {
  console.log(`\nValidating "${scenario.name}" (${scenario.id})...`);

  // Structural checks.
  check_structure(scenario);
  ok('structural checks OK');

  // Base simulation (with hardware expansion at default count).
  const result = check_simulation('base', scenario);
  if (result) {
    ok(`simulation OK (${result.max_tick} ticks, ${
        result.op_count} ops retired)`);

    // Annotation ticks within range.
    if (scenario.annotations) {
      for (const ann of scenario.annotations) {
        if (ann.tick > result.max_tick) {
          error(
              scenario,
              `annotation at t=${ann.tick} exceeds final tick ${
                  result.max_tick}`);
        }
      }
    }
  }

  // Identity check.
  const identity = scaleScenario(scenario, 1, 1);
  if (identity !== scenario) {
    error(scenario, 'scaleScenario(base, 1, 1) is not reference-equal to base');
  } else {
    ok('identity OK');
  }

  // Hardware count scaling (only if scalable hardware exists).
  const scalable_hw = getScalableHardware(scenario);
  if (scalable_hw) {
    for (const hw_count of [1, 2, 3]) {
      if (hw_count === scalable_hw.count) continue;
      const label = `hw=${hw_count}`;
      const expanded = expandHardware(scenario, hw_count);
      const sr = check_simulation(label, expanded);
      if (sr) {
        ok(`${label}: OK (${sr.max_tick} ticks, ${sr.op_count} ops)`);
      }
    }
  } else {
    ok('hw scaling: not supported (no scalable hardware)');
  }

  // Width-only scaling (with hardware expansion).
  for (const width of [2, 3]) {
    const label = `${width}x1`;
    const scaled = scaleScenario(scenario, width, 1);
    if (check_scaled_structure(label, scaled)) {
      const sr = check_simulation(label, scaled);
      if (sr) {
        ok(`${label}: OK (${sr.max_tick} ticks, ${sr.op_count} ops)`);
      }
    }
  }

  // Depth scaling (only if chain_through).
  if (supportsDepth(scenario)) {
    for (const [width, depth] of [[1, 2], [2, 2], [1, 3]]) {
      const label = `${width}x${depth}`;
      const scaled = scaleScenario(scenario, width, depth);
      if (check_scaled_structure(label, scaled)) {
        const sr = check_simulation(label, scaled);
        if (sr) {
          ok(`${label}: OK (${sr.max_tick} ticks, ${sr.op_count} ops)`);
        }
      }
    }
  } else {
    ok('depth: not supported (no chain_through or iteration_deps)');
  }
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

console.log(
    `\nValidation complete: ${errors} error(s), ${warnings} warning(s)`);
process.exit(errors > 0 ? 1 : 0);
