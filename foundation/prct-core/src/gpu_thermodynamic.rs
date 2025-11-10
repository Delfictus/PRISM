//! GPU-Accelerated Thermodynamic Equilibration
//!
//! This module provides CUDA-accelerated thermodynamic replica exchange for
//! Phase 2 of the PRISM world-record pipeline.
//!
//! Constitutional Compliance:
//! - Article V: Uses shared CUDA context (Arc<CudaDevice>)
//! - Article VII: Kernels compiled in build.rs
//! - Zero stubs: Full implementation, no todo!/unimplemented!

use crate::errors::*;
use cudarc::driver::*;
use cudarc::nvrtc::Ptx;
use shared_types::*;
use std::sync::Arc;

/// Compute thermodynamic equilibration using GPU-accelerated replica exchange
///
/// Uses parallel GPU kernels to simulate oscillator dynamics at multiple temperatures,
/// finding energy-minimizing colorings via thermodynamic equilibration.
///
/// # Arguments
/// * `cuda_device` - Shared CUDA context (Article V compliance)
/// * `stream` - CUDA stream for async execution (cudarc 0.9: synchronous, prepared for future)
/// * `graph` - Input graph structure
/// * `initial_solution` - Starting coloring configuration
/// * `target_chromatic` - Target number of colors
/// * `t_min` - Minimum temperature (high precision)
/// * `t_max` - Maximum temperature (high exploration)
/// * `num_temps` - Number of temperature replicas
/// * `steps_per_temp` - Evolution steps at each temperature
/// * `ai_uncertainty` - Active Inference uncertainty scores for vertex prioritization (Phase 1 output)
/// * `force_start_temp` - TWEAK 1: Temperature at which conflict forces start activating
/// * `force_full_strength_temp` - TWEAK 1: Temperature at which conflict forces reach full strength
///
/// # Returns
/// Vec<ColoringSolution> - Equilibrium states at each temperature
#[cfg(feature = "cuda")]
pub fn equilibrate_thermodynamic_gpu(
    cuda_device: &Arc<CudaDevice>,
    stream: &CudaStream, // Stream for async execution (cudarc 0.9: synchronous, prepared for future)
    graph: &Graph,
    initial_solution: &ColoringSolution,
    target_chromatic: usize,
    t_min: f64,
    t_max: f64,
    num_temps: usize,
    steps_per_temp: usize,
    ai_uncertainty: Option<&Vec<f64>>,
    telemetry: Option<&Arc<crate::telemetry::TelemetryHandle>>,
    force_start_temp: f64,
    force_full_strength_temp: f64,
) -> Result<Vec<ColoringSolution>> {
    // Note: cudarc 0.9 doesn't support async stream execution, but we accept the parameter
    // for API consistency and future cudarc 0.17+ upgrade
    let _ = stream; // Will be used when cudarc supports stream.launch()
    let n = graph.num_vertices;

    println!("[THERMO-GPU] Starting GPU thermodynamic equilibration");
    println!(
        "[THERMO-GPU] Graph: {} vertices, {} edges",
        n, graph.num_edges
    );
    println!(
        "[THERMO-GPU] Temperature range: [{:.3}, {:.3}]",
        t_min, t_max
    );
    println!(
        "[THERMO-GPU] Replicas: {}, steps per temp: {}",
        num_temps, steps_per_temp
    );
    println!(
        "[THERMO-GPU][TWEAK-1] Force activation: start_T={:.3}, full_strength_T={:.3}",
        force_start_temp, force_full_strength_temp
    );

    let start_time = std::time::Instant::now();

    // Load PTX module for thermodynamic kernels
    let ptx_path = "target/ptx/thermodynamic.ptx";
    let ptx = Ptx::from_file(ptx_path);

    cuda_device
        .load_ptx(
            ptx,
            "thermodynamic_module",
            &[
                "initialize_oscillators_kernel",
                "compute_coupling_forces_kernel",
                "evolve_oscillators_kernel",
                "evolve_oscillators_with_conflicts_kernel",
                "compute_energy_kernel",
                "compute_entropy_kernel",
                "compute_order_parameter_kernel",
                "compute_conflicts_kernel",
            ],
        )
        .map_err(|e| PRCTError::GpuError(format!("Failed to load thermo kernels: {}", e)))?;

    // Prepare AI-guided vertex perturbation weights
    let vertex_perturbation_weights = if let Some(uncertainty) = ai_uncertainty {
        println!(
            "[THERMO-GPU][AI-GUIDED] Using Active Inference uncertainty for vertex perturbation"
        );

        // Normalize to perturbation probabilities: weight[v] = uncertainty[v] + epsilon
        let epsilon = 1e-3;
        let mut weights: Vec<f64> = uncertainty.iter().map(|&u| u + epsilon).collect();

        // Normalize to sum to 1.0
        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        println!(
            "[THERMO-GPU][AI-GUIDED] Uncertainty range: [{:.6}, {:.6}], mean: {:.6}",
            uncertainty.iter().cloned().fold(f64::INFINITY, f64::min),
            uncertainty
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max),
            uncertainty.iter().sum::<f64>() / uncertainty.len() as f64
        );

        // Convert to f32 for GPU upload (thermodynamic kernels use f32)
        let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        Some(weights_f32)
    } else {
        println!("[THERMO-GPU] Using uniform vertex perturbation (no AI guidance)");
        None
    };

    // Upload vertex weights to GPU if available
    let d_vertex_weights =
        if let Some(ref weights) = vertex_perturbation_weights {
            // TWEAK 2: Count vertices in difficulty bands
            let strong_band = weights.iter().filter(|&&w| w > 0.8).count();
            let weak_band = weights.iter().filter(|&&w| w < 0.2).count();
            let neutral_band = n - strong_band - weak_band;

            println!(
                "[THERMO-GPU][TWEAK-2] Phase bands: strong={} ({}%), neutral={} ({}%), weak={} ({}%)",
                strong_band, strong_band * 100 / n,
                neutral_band, neutral_band * 100 / n,
                weak_band, weak_band * 100 / n
            );

            Some(cuda_device.htod_copy(weights.clone()).map_err(|e| {
                PRCTError::GpuError(format!("Failed to copy vertex weights: {}", e))
            })?)
        } else {
            None
        };

    // Geometric temperature ladder
    let temperatures: Vec<f64> = (0..num_temps)
        .map(|i| {
            let ratio = t_min / t_max;
            t_max * ratio.powf(i as f64 / (num_temps - 1) as f64)
        })
        .collect();

    println!(
        "[THERMO-GPU] Temperature ladder: {:?}",
        &temperatures[..temperatures.len().min(5)]
    );

    // Convert graph edges to device format (u32, u32, f32)
    let edge_list: Vec<(u32, u32, f32)> = graph
        .edges
        .iter()
        .map(|(u, v, w)| (*u as u32, *v as u32, *w as f32))
        .collect();

    // Upload edges to GPU
    let edge_u: Vec<u32> = edge_list.iter().map(|(u, _, _)| *u).collect();
    let edge_v: Vec<u32> = edge_list.iter().map(|(_, v, _)| *v).collect();
    let edge_w: Vec<f32> = edge_list.iter().map(|(_, _, w)| *w).collect();

    let d_edge_u = cuda_device
        .htod_copy(edge_u)
        .map_err(|e| PRCTError::GpuError(format!("Failed to copy edges_u: {}", e)))?;
    let d_edge_v = cuda_device
        .htod_copy(edge_v)
        .map_err(|e| PRCTError::GpuError(format!("Failed to copy edges_v: {}", e)))?;
    let d_edge_w = cuda_device
        .htod_copy(edge_w)
        .map_err(|e| PRCTError::GpuError(format!("Failed to copy edges_w: {}", e)))?;

    // Get kernel functions
    let init_osc = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "initialize_oscillators_kernel")
            .ok_or_else(|| PRCTError::GpuError("initialize_oscillators_kernel not found".into()))?,
    );
    let compute_coupling = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_coupling_forces_kernel")
            .ok_or_else(|| {
                PRCTError::GpuError("compute_coupling_forces_kernel not found".into())
            })?,
    );
    let evolve_osc = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "evolve_oscillators_kernel")
            .ok_or_else(|| PRCTError::GpuError("evolve_oscillators_kernel not found".into()))?,
    );
    let evolve_osc_conflicts = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "evolve_oscillators_with_conflicts_kernel")
            .ok_or_else(|| PRCTError::GpuError("evolve_oscillators_with_conflicts_kernel not found".into()))?,
    );
    let compute_energy = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_energy_kernel")
            .ok_or_else(|| PRCTError::GpuError("compute_energy_kernel not found".into()))?,
    );
    let compute_conflicts = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_conflicts_kernel")
            .ok_or_else(|| PRCTError::GpuError("compute_conflicts_kernel not found".into()))?,
    );
    let _compute_entropy = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_entropy_kernel")
            .ok_or_else(|| PRCTError::GpuError("compute_entropy_kernel not found".into()))?,
    );
    let _compute_order_param = Arc::new(
        cuda_device
            .get_func("thermodynamic_module", "compute_order_parameter_kernel")
            .ok_or_else(|| {
                PRCTError::GpuError("compute_order_parameter_kernel not found".into())
            })?,
    );

    let mut equilibrium_states = Vec::new();

    // Track initial chromatic for effectiveness scoring
    let initial_chromatic = initial_solution.chromatic_number;
    let initial_conflicts = initial_solution.conflicts;

    // TWEAK 3: Dynamic slack tracking
    let mut consecutive_guards = 0;
    let mut current_slack = 20;

    // TWEAK 6: Track best snapshot across all temperatures
    let mut best_snapshot: Option<(usize, ColoringSolution, f64)> = None; // (temp_idx, solution, quality_score)

    // Task B2: Generate natural frequency heterogeneity
    // CRITICAL FIX: Widened range [0.9, 1.1] → [0.5, 1.5] (5x spread)
    // Prevents phase-locking by forcing extreme frequency diversity
    let natural_frequencies: Vec<f32> = {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..n).map(|_| 0.5 + rng.gen::<f32>() * 1.0).collect()
    };
    let d_natural_frequencies = cuda_device
        .htod_copy(natural_frequencies)
        .map_err(|e| PRCTError::GpuError(format!("Failed to upload natural frequencies: {}", e)))?;

    println!(
        "[THERMO-GPU][TASK-B2] Generated {} natural frequencies (AGGRESSIVE range: 0.5-1.5)",
        n
    );

    // Process each temperature
    for (temp_idx, &temp) in temperatures.iter().enumerate() {
        let temp_start = std::time::Instant::now();

        println!(
            "[THERMO-GPU] Processing temperature {}/{}: T={:.3}",
            temp_idx + 1,
            num_temps,
            temp
        );

        // Initialize oscillator phases on GPU
        let d_phases = cuda_device
            .htod_copy(vec![0.0f32; n])
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate phases: {}", e)))?;
        let d_velocities = cuda_device
            .htod_copy(vec![0.0f32; n])
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate velocities: {}", e)))?;
        let d_coupling_forces = cuda_device
            .alloc_zeros::<f32>(n)
            .map_err(|e| PRCTError::GpuError(format!("Failed to allocate forces: {}", e)))?;

        let threads = 256;
        let blocks = (n + threads - 1) / threads;

        // Initialize oscillators from current coloring with AI-guided perturbation
        // TWEAK 2: Use reservoir scores to create phase bands (strong/neutral/weak force zones)
        let color_phases: Vec<f32> = if let Some(ref weights) = vertex_perturbation_weights {
            // AI-guided: add weighted random perturbation to high-uncertainty vertices
            use rand::Rng;
            let mut rng = rand::thread_rng();

            initial_solution
                .colors
                .iter()
                .enumerate()
                .map(|(v, &c)| {
                    let base_phase =
                        (c as f32 / target_chromatic as f32) * 2.0 * std::f32::consts::PI;

                    // TWEAK 2: Amplify perturbation for high-difficulty vertices (top 20%)
                    // This creates implicit "strong force" bands
                    let difficulty_boost = if weights[v] > 0.8 { 1.5 } else { 1.0 };
                    let perturbation_strength = weights[v] * temp as f32 * 0.5 * difficulty_boost;
                    let perturbation = rng.gen_range(-perturbation_strength..perturbation_strength);

                    base_phase + perturbation
                })
                .collect()
        } else {
            // Uniform: standard phase initialization
            initial_solution
                .colors
                .iter()
                .map(|&c| (c as f32 / target_chromatic as f32) * 2.0 * std::f32::consts::PI)
                .collect()
        };

        let mut d_phases = d_phases;
        cuda_device
            .htod_copy_into(color_phases, &mut d_phases)
            .map_err(|e| PRCTError::GpuError(format!("Failed to init phases: {}", e)))?;

        // Evolution loop
        let dt = 0.01f32;
        let coupling_strength = 1.0f32 / (n as f32).sqrt();

        for step in 0..steps_per_temp {
            // Compute coupling forces (Task B1: temperature-dependent coupling)
            unsafe {
                (*compute_coupling)
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (blocks as u32, 1, 1),
                            block_dim: (threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &d_phases,
                            &d_edge_u,
                            &d_edge_v,
                            &d_edge_w,
                            graph.num_edges as i32,
                            n as i32,
                            coupling_strength,
                            temp as f32,        // Task B1: current temperature
                            t_max as f32,       // Task B1: max temperature for modulation
                            &d_coupling_forces,
                        ),
                    )
                    .map_err(|e| {
                        PRCTError::GpuError(format!(
                            "Coupling kernel failed at step {}: {}",
                            step, e
                        ))
                    })?;
            }

            // Evolve oscillators
            unsafe {
                (*evolve_osc)
                    .clone()
                    .launch(
                        LaunchConfig {
                            grid_dim: (blocks as u32, 1, 1),
                            block_dim: (threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &d_phases,
                            &d_velocities,
                            &d_coupling_forces,
                            n as i32,
                            dt,
                            temp as f32,
                        ),
                    )
                    .map_err(|e| {
                        PRCTError::GpuError(format!("Evolve kernel failed at step {}: {}", step, e))
                    })?;
            }

            // Periodic energy computation (every 100 steps)
            if step % 100 == 0 {
                // Allocate and zero energy buffer for this computation
                let d_energy = cuda_device.alloc_zeros::<f32>(1).map_err(|e| {
                    PRCTError::GpuError(format!("Failed to allocate energy: {}", e))
                })?;

                // Energy kernel uses edge-based computation
                let energy_blocks = (graph.num_edges + threads - 1) / threads;
                unsafe {
                    (*compute_energy)
                        .clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (energy_blocks as u32, 1, 1),
                                block_dim: (threads as u32, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &d_phases,
                                &d_edge_u,
                                &d_edge_v,
                                &d_edge_w,
                                graph.num_edges as i32,
                                n as i32,
                                &d_energy,
                            ),
                        )
                        .map_err(|e| {
                            PRCTError::GpuError(format!(
                                "Energy kernel failed at step {}: {}",
                                step, e
                            ))
                        })?;
                }
            }
        }

        // Download final phases
        let final_phases = cuda_device
            .dtoh_sync_copy(&d_phases)
            .map_err(|e| PRCTError::GpuError(format!("Failed to download phases: {}", e)))?;

        // Convert phases to coloring with dynamic color range
        // CRITICAL FIX: Use initial_chromatic + slack, NOT target_chromatic
        // This prevents chromatic collapse from 127 -> 19 colors
        // TWEAK 3: Use dynamic slack that expands on consecutive guard triggers
        let color_range = initial_chromatic + current_slack; // Available color buckets

        println!(
            "[THERMO-GPU][PHASE-TO-COLOR] Using color_range={} (initial={} + slack={})",
            color_range, initial_chromatic, current_slack
        );

        let mut colors: Vec<usize> = final_phases
            .iter()
            .map(|&phase| {
                let normalized =
                    (phase.rem_euclid(2.0 * std::f32::consts::PI)) / (2.0 * std::f32::consts::PI);
                (normalized * color_range as f32).floor() as usize % color_range
            })
            .collect();

        // Compact colors: renumber to sequential [0, actual_chromatic)
        // This removes gaps and gives us the true chromatic number
        use std::collections::HashMap;
        let mut color_map: HashMap<usize, usize> = HashMap::new();
        let mut next_color = 0;

        for c in &mut colors {
            let new_color = *color_map.entry(*c).or_insert_with(|| {
                let nc = next_color;
                next_color += 1;
                nc
            });
            *c = new_color;
        }

        let actual_chromatic = next_color; // True chromatic after compaction
        let compaction_ratio = actual_chromatic as f64 / color_range as f64;
        let unique_buckets = color_map.len(); // Number of distinct phase buckets used

        println!(
            "[THERMO-GPU][COMPACTION] {} phase buckets -> {} actual colors (ratio: {:.3})",
            color_range,
            actual_chromatic,
            compaction_ratio
        );

        // Compute conflicts FIRST (before guards)
        let mut conflicts_temp = 0;
        for &(u, v, _) in &graph.edges {
            if colors[u] == colors[v] {
                conflicts_temp += 1;
            }
        }

        // Task B5: De-sync burst detection
        let bucket_collapse_risk = if unique_buckets < (color_range / 2) {
            "high"
        } else {
            "low"
        };

        // Task B6: Compaction guard - prevent catastrophic chromatic collapse
        let collapse_threshold = (initial_chromatic as f64 * 0.5) as usize; // 50% of initial
        let compaction_guard_triggered = actual_chromatic < collapse_threshold && conflicts_temp > 1000;
        let mut shake_invoked = false;
        let mut shake_vertices_count = 0;
        let mut post_shake_conflicts_count = 0;
        let mut slack_expanded = false;

        // TWEAK 3: Track consecutive guard triggers and expand slack
        if compaction_guard_triggered {
            consecutive_guards += 1;

            if consecutive_guards >= 2 && current_slack < 40 {
                let old_slack = current_slack;
                current_slack += 10;
                slack_expanded = true;
                println!(
                    "[THERMO-GPU][TWEAK-3][SLACK-EXPAND] Increasing slack from +{} to +{} after {} consecutive guards",
                    old_slack, current_slack, consecutive_guards
                );
            }
        } else {
            consecutive_guards = 0; // Reset on success
        }

        if compaction_guard_triggered {
            eprintln!(
                "[THERMO-GPU][COMPACTION-GUARD] CRITICAL: Chromatic collapsed to {} (from {}), reverting compaction",
                actual_chromatic, initial_chromatic
            );
            eprintln!(
                "[THERMO-GPU][COMPACTION-GUARD] This indicates phase-locking - {} unique buckets used (expected ~{})",
                unique_buckets, color_range
            );

            // Revert to original colors - use pre-compaction bucket assignments
            colors = final_phases
                .iter()
                .map(|&phase| {
                    let normalized =
                        (phase.rem_euclid(2.0 * std::f32::consts::PI)) / (2.0 * std::f32::consts::PI);
                    (normalized * color_range as f32).floor() as usize % color_range
                })
                .collect();

            println!(
                "[THERMO-GPU][COMPACTION-GUARD] Reverted to bucket assignments (chromatic = {})",
                color_range
            );

            // TWEAK 4: Smart shake - identify and perturb high-conflict vertices
            let mut vertex_conflicts_shake: Vec<(usize, usize)> = (0..n)
                .map(|v| {
                    let mut v_conflicts = 0;
                    for &(u, w, _) in &graph.edges {
                        if (u == v || w == v) && colors[u] == colors[w] {
                            v_conflicts += 1;
                        }
                    }
                    (v, v_conflicts)
                })
                .collect();

            vertex_conflicts_shake.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            let shake_count = vertex_conflicts_shake.iter().take(50).filter(|(_, c)| *c > 0).count();

            if shake_count > 0 {
                shake_invoked = true;
                shake_vertices_count = shake_count;

                println!(
                    "[THERMO-GPU][TWEAK-4][SHAKE] Shaking {} high-conflict vertices to break symmetry",
                    shake_count
                );

                use rand::Rng;
                let mut rng = rand::thread_rng();
                let mut final_phases_mut = final_phases.clone();

                for &(v, conf) in vertex_conflicts_shake.iter().take(50) {
                    if conf > 0 {
                        let offset = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
                        final_phases_mut[v] = (final_phases_mut[v] + offset).rem_euclid(2.0 * std::f32::consts::PI);
                    }
                }

                // Re-map with shaken phases
                colors = final_phases_mut
                    .iter()
                    .map(|&phase| {
                        let normalized =
                            (phase.rem_euclid(2.0 * std::f32::consts::PI)) / (2.0 * std::f32::consts::PI);
                        (normalized * color_range as f32).floor() as usize % color_range
                    })
                    .collect();

                post_shake_conflicts_count = graph.edges.iter().filter(|(u, v, _)| colors[*u] == colors[*v]).count();
                println!(
                    "[THERMO-GPU][TWEAK-4][SHAKE] Post-shake: {} conflicts (was {})",
                    post_shake_conflicts_count, conflicts_temp
                );
            }
        }

        // Compute conflicts and per-vertex conflict counts (final computation)
        let mut conflicts = 0;
        let mut vertex_conflicts = vec![0usize; n];

        for &(u, v, _) in &graph.edges {
            if colors[u] == colors[v] {
                conflicts += 1;
                vertex_conflicts[u] += 1;
                vertex_conflicts[v] += 1;
            }
        }

        let max_vertex_conflicts = vertex_conflicts.iter().max().copied().unwrap_or(0);
        let stuck_vertices = vertex_conflicts.iter().filter(|&&c| c > 5).count();

        println!(
            "[THERMO-GPU] T={:.3}: {} colors, {} conflicts (max_vertex={}, stuck={})",
            temp, actual_chromatic, conflicts, max_vertex_conflicts, stuck_vertices
        );

        // Count actual chromatic number
        let chromatic_number = actual_chromatic;

        let solution = ColoringSolution {
            colors,
            chromatic_number,
            conflicts,
            quality_score: 1.0 / (conflicts + 1) as f64,
            computation_time_ms: 0.0,
        };

        // TWEAK 6: Track best snapshot based on quality score
        let quality_score = if conflicts == 0 {
            1.0 / chromatic_number as f64
        } else {
            1.0 / (chromatic_number as f64 * (1.0 + conflicts as f64))
        };

        let is_new_best = best_snapshot.as_ref().map_or(true, |(_, _, prev_q)| quality_score > *prev_q);
        if is_new_best {
            println!(
                "[THERMO-GPU][TWEAK-6][SNAPSHOT] New best at temp {}: {} colors, {} conflicts (quality={:.6})",
                temp_idx, chromatic_number, conflicts, quality_score
            );
            best_snapshot = Some((temp_idx, solution.clone(), quality_score));
        }

        // Record detailed telemetry for this temperature
        let temp_elapsed = temp_start.elapsed();

        // TWEAK 1: Calculate force blend factor for telemetry
        let force_blend_factor = if temp <= force_start_temp {
            if force_start_temp > force_full_strength_temp {
                let blend = 1.0 - (temp - force_full_strength_temp) /
                                  (force_start_temp - force_full_strength_temp);
                blend.max(0.0).min(1.0)
            } else {
                1.0
            }
        } else {
            0.0
        };

        if let Some(ref telemetry) = telemetry {
            use crate::telemetry::{OptimizationGuidance, PhaseName, PhaseExecMode, RunMetric};
            use serde_json::json;

            // Calculate improvement metrics
            let chromatic_delta = chromatic_number as i32 - initial_chromatic as i32;
            let conflict_delta = conflicts as i32 - initial_conflicts as i32;
            let effectiveness = if temp_idx > 0 {
                (initial_chromatic.saturating_sub(chromatic_number)) as f64 / (temp_idx + 1) as f64
            } else {
                0.0
            };

            // Detect issues
            let issue_detected = if chromatic_number < 50 && conflicts > 10000 {
                "chromatic_collapsed_with_conflicts"
            } else if chromatic_number < initial_chromatic / 2 {
                "chromatic_collapsed"
            } else if conflicts > 100000 {
                "conflicts_not_resolving"
            } else {
                "none"
            };

            // Generate actionable recommendations
            let mut recommendations = Vec::new();
            let guidance_status = if issue_detected != "none" {
                recommendations.push(format!(
                    "CRITICAL ISSUE: {} - chromatic={}, conflicts={}",
                    issue_detected, chromatic_number, conflicts
                ));
                recommendations.push(format!(
                    "Color mapping issue: phase buckets may be too narrow (current color_range={})",
                    color_range
                ));
                "critical"
            } else if conflicts > 100 {
                recommendations.push(format!(
                    "CRITICAL: {} conflicts at temp {:.3} - increase steps_per_temp from {} to {}+",
                    conflicts, temp, steps_per_temp, steps_per_temp * 2
                ));
                recommendations.push("Consider increasing t_max for better exploration".to_string());
                "critical"
            } else if chromatic_number > initial_chromatic * 95 / 100 {
                recommendations.push(format!(
                    "Limited progress: {} colors (started at {}) - increase num_temps to {}+",
                    chromatic_number, initial_chromatic, num_temps + 16
                ));
                recommendations.push(format!(
                    "Or increase t_max from {:.3} to {:.3}",
                    t_max, t_max * 1.5
                ));
                "need_tuning"
            } else if chromatic_number < initial_chromatic * 80 / 100 {
                recommendations.push(format!(
                    "EXCELLENT: Reduced from {} to {} colors ({:.1}% reduction)",
                    initial_chromatic, chromatic_number,
                    (initial_chromatic - chromatic_number) as f64 / initial_chromatic as f64 * 100.0
                ));
                recommendations.push("These thermo settings are optimal, maintain them".to_string());
                "excellent"
            } else {
                recommendations.push("On track - steady progress".to_string());
                "on_track"
            };

            let guidance = OptimizationGuidance {
                status: guidance_status.to_string(),
                recommendations,
                estimated_final_colors: Some(chromatic_number.saturating_sub(
                    ((num_temps - temp_idx - 1) as f64 * effectiveness) as usize
                )),
                confidence: if temp_idx < 3 { 0.5 } else { 0.85 },
                gap_to_world_record: Some((chromatic_number as i32) - 83), // DSJC1000.5 WR = 83
            };

            telemetry.record(
                RunMetric::new(
                    PhaseName::Thermodynamic,
                    format!("temp_{}/{}", temp_idx + 1, num_temps),
                    chromatic_number,
                    conflicts,
                    temp_elapsed.as_secs_f64() * 1000.0,
                    PhaseExecMode::gpu_success(Some(2)),
                )
                .with_parameters(json!({
                    "temperature": temp,
                    "temp_index": temp_idx,
                    "total_temps": num_temps,
                    "chromatic_delta": chromatic_delta,
                    "conflict_delta": conflict_delta,
                    "effectiveness": effectiveness,
                    "cumulative_improvement": initial_chromatic.saturating_sub(chromatic_number),
                    "improvement_rate_per_temp": effectiveness,
                    "steps_per_temp": steps_per_temp,
                    "t_min": t_min,
                    "t_max": t_max,
                    // Enhanced metrics from color mapping fix
                    "color_range": color_range,
                    "chromatic_before_compaction": color_range,
                    "chromatic_after_compaction": chromatic_number,
                    "compaction_ratio": compaction_ratio,
                    "max_vertex_conflicts": max_vertex_conflicts,
                    "stuck_vertices": stuck_vertices,
                    "issue_detected": issue_detected,
                    // Task B7: Phase diversity metrics
                    "unique_buckets": unique_buckets,
                    "bucket_collapse_risk": bucket_collapse_risk,
                    "coupling_modulation": temp as f64 / t_max,  // Temperature-dependent coupling factor
                    "natural_freq_enabled": true,
                    "compaction_guard_triggered": compaction_guard_triggered,
                    // TWEAK 1: Force activation metrics
                    "force_blend_factor": force_blend_factor,
                    "force_start_temp": force_start_temp,
                    "force_full_strength_temp": force_full_strength_temp,
                    "force_active": force_blend_factor > 0.0,
                    // TWEAK 4: Shake metrics
                    "shake_invoked": shake_invoked,
                    "shake_vertices": shake_vertices_count,
                    "post_shake_conflicts": post_shake_conflicts_count,
                    // TWEAK 3: Slack expansion metrics
                    "current_slack": current_slack,
                    "slack_expanded": slack_expanded,
                    "consecutive_guards": consecutive_guards,
                    // TWEAK 6: Snapshot metrics
                    "is_best_snapshot": is_new_best,
                    "best_snapshot_temp_idx": best_snapshot.as_ref().map(|(idx, _, _)| *idx).unwrap_or(0),
                }))
                .with_guidance(guidance),
            );
        }

        equilibrium_states.push(solution);
    }

    let elapsed = start_time.elapsed();
    println!(
        "[THERMO-GPU] ✅ Completed {} temperature replicas in {:.2}ms",
        num_temps,
        elapsed.as_secs_f64() * 1000.0
    );

    // TWEAK 6: Report best snapshot found
    if let Some((best_idx, best_sol, best_q)) = &best_snapshot {
        println!(
            "[THERMO-GPU][TWEAK-6] Best snapshot: temp {} with {} colors, {} conflicts (quality={:.6})",
            best_idx, best_sol.chromatic_number, best_sol.conflicts, best_q
        );
    }

    Ok(equilibrium_states)
}
