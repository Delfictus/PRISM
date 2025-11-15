//! World Record Attempt for DSJC1000.5
//!
//! Uses DSATUR with warm start and aggressive thermodynamic settings
//! OR WorldRecordPipeline with full GPU acceleration (with --features cuda)
//! Target: ≤82 colors (world record)

use anyhow::Result;
use ndarray::Array2;
use prct_core::{parse_dimacs_file, ColoringSolution, DSaturSolver};
use std::env;
use std::path::Path;
use std::time::Instant;
use std::thread;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
use prct_core::world_record_pipeline::{WorldRecordConfig, WorldRecordPipeline};

#[cfg(feature = "cuda")]
use prct_core::gpu::device_topology;

fn main() -> Result<()> {
    println!(
        r#"
╔══════════════════════════════════════════════════════════════════╗
║          PRISM-AI WORLD RECORD ATTEMPT: DSJC1000.5              ║
║                                                                  ║
║  Target: ≤82 colors (Current World Record)                      ║
║                                                                  ║
║  Techniques:                                                     ║
║    • DSATUR with Warm Start Auto-Adjustment                    ║
║    • Aggressive Thermodynamic Replica Exchange                 ║
║    • Branch-and-Bound Backtracking                             ║
║    • 96 Temperature Replicas with 12K steps/temp               ║
║    • 8-GPU RunPod Configuration (with CUDA)                    ║
║                                                                  ║
║  Configuration: runpod_8gpu.v1.1.toml                           ║
╚══════════════════════════════════════════════════════════════════╝
"#
    );

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let config_path = if args.len() > 1 {
        &args[1]
    } else {
        "foundation/prct-core/configs/runpod_8gpu.v1.1.toml"
    };

    let graph_path = if args.len() > 2 {
        &args[2]
    } else {
        "benchmarks/dimacs/DSJC1000.5.col"
    };

    println!("\n📊 Loading configuration and graph...");
    println!("   Config: {}", config_path);
    println!("   Graph:  {}", graph_path);

    // Load graph
    let (num_vertices, edges) = parse_dimacs_file(Path::new(graph_path))?;

    println!("\n✅ Graph loaded:");
    println!("   Vertices: {}", num_vertices);
    println!("   Edges: {}", edges.len());
    let density = (2.0 * edges.len() as f64) / (num_vertices as f64 * (num_vertices - 1) as f64);
    println!("   Density: {:.1}%", density * 100.0);
    println!("   Best known: 83 colors (world record)");
    println!();

    // Check if GPU pipeline should be used
    #[cfg(feature = "cuda")]
    {
        if std::env::var("PRISM_USE_DSATUR").is_err() {
            println!("🚀 Using GPU WorldRecordPipeline (set PRISM_USE_DSATUR=1 to use DSATUR instead)");
            return run_gpu_pipeline(config_path, &edges, num_vertices);
        }
    }

    // Fall back to DSATUR approach
    println!("🎯 Using DSATUR solver with warm start");
    run_dsatur_solver(edges, num_vertices)
}

#[cfg(feature = "cuda")]
fn run_gpu_pipeline(config_path: &str, edges: &[(usize, usize)], num_vertices: usize) -> Result<()> {
    use shared_types::{Graph, KuramotoState};

    // Convert to Graph structure
    let num_edges = edges.len();
    let mut adjacency = vec![false; num_vertices * num_vertices];
    for &(u, v) in edges {
        adjacency[u * num_vertices + v] = true;
        adjacency[v * num_vertices + u] = true;
    }

    let edges_weighted: Vec<(usize, usize, f32)> = edges.iter()
        .map(|&(u, v)| (u, v, 1.0))
        .collect();

    let graph = Graph {
        num_vertices,
        num_edges,
        edges: edges_weighted,
        adjacency,
        coordinates: None,
    };

    // Detect available GPUs
    println!("🚀 Detecting GPUs...");
    let device_filter = vec!["cuda:*".to_string()];
    let devices = device_topology::discover(&device_filter)?;

    println!("✅ Found {} GPU(s):", devices.len());
    for dev in &devices {
        println!("   {} - {} MB", dev.name, dev.memory_mb);
    }
    println!();

    let num_gpus = devices.len();
    let use_multi_gpu = num_gpus > 1 && std::env::var("PRISM_SINGLE_GPU").is_err();

    if use_multi_gpu {
        println!("🚀 Multi-GPU mode: Running {} parallel replicas", num_gpus);
    } else {
        println!("🎯 Single-GPU mode: Using GPU 0");
    }
    println!();

    // Initialize Kuramoto oscillators for graph dynamics
    println!("🌊 Initializing Kuramoto oscillators...");
    let mut phases = vec![0.0; graph.num_vertices];
    for (i, phase) in phases.iter_mut().enumerate() {
        *phase = (i as f64 * 2.0 * std::f64::consts::PI) / graph.num_vertices as f64;
    }

    let natural_frequencies = vec![1.0; graph.num_vertices];
    let n = graph.num_vertices;
    let mut coupling_matrix = vec![0.0; n * n];

    // Set coupling based on adjacency
    for &(u, v, _) in &graph.edges {
        coupling_matrix[u * n + v] = 1.0;
        coupling_matrix[v * n + u] = 1.0;
    }

    let kuramoto = KuramotoState {
        phases,
        natural_frequencies,
        coupling_matrix,
        order_parameter: 0.5,
        mean_phase: 0.0,
    };

    // Load configuration
    println!("📂 Loading configuration from: {}", config_path);
    let config = WorldRecordConfig::from_file(config_path)?;
    println!("✅ Configuration loaded and validated");
    println!();

    println!("🎯 Pipeline Configuration:");
    println!("   Target: {} colors", config.target_chromatic);
    println!("   Max Runtime: {:.1} hours", config.max_runtime_hours);
    println!("   Workers: {}", config.num_workers);
    println!("   GPU Reservoir: {}", if config.use_reservoir_prediction { "✅" } else { "❌" });
    println!("   Active Inference: {}", if config.use_active_inference { "✅" } else { "❌" });
    println!("   ADP Q-Learning: {}", if config.use_adp_learning { "✅" } else { "❌" });
    println!("   Thermodynamic: {}", if config.use_thermodynamic_equilibration { "✅" } else { "❌" });
    println!("   Quantum-Classical: {}", if config.use_quantum_classical_hybrid { "✅" } else { "❌" });
    println!("   Multi-Scale: {}", if config.use_multiscale_analysis { "✅" } else { "❌" });
    println!("   Ensemble Consensus: {}", if config.use_ensemble_consensus { "✅" } else { "❌" });
    println!();

    // Run world record attempt
    println!("🏁 Starting World Record Attempt...");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let start = std::time::Instant::now();

    let result = if use_multi_gpu {
        // Multi-GPU: spawn thread per GPU
        let handles: Vec<_> = (0..num_gpus).map(|gpu_id| {
            let graph_clone = graph.clone();
            let kuramoto_clone = kuramoto.clone();
            let config_clone = config.clone();

            thread::spawn(move || -> Result<_> {
                let cuda_device = CudaDevice::new(gpu_id)?;
                let run_id = format!("dsjc1000_gpu{}", gpu_id);
                let pipeline = WorldRecordPipeline::new(config_clone, cuda_device)?;
                let mut pipeline = pipeline.with_telemetry(&run_id)?;
                pipeline.optimize_world_record(&graph_clone, &kuramoto_clone)
                    .map_err(|e| anyhow::anyhow!("GPU {} error: {}", gpu_id, e))
            })
        }).collect();

        // Wait for all GPUs and take best result
        let mut best_result = None;
        for (gpu_id, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(Ok(result)) => {
                    println!("✅ GPU {} finished: {} colors, {} conflicts",
                             gpu_id, result.chromatic_number, result.conflicts);
                    if best_result.is_none() ||
                       result.chromatic_number < best_result.as_ref().unwrap().chromatic_number {
                        best_result = Some(result);
                    }
                }
                Ok(Err(e)) => println!("❌ GPU {} failed: {}", gpu_id, e),
                Err(_) => println!("❌ GPU {} thread panicked", gpu_id),
            }
        }

        best_result.ok_or_else(|| anyhow::anyhow!("All GPU threads failed"))?
    } else {
        // Single-GPU
        let cuda_device = CudaDevice::new(0)?;
        let run_id = format!("dsjc1000_{}", chrono::Local::now().format("%Y%m%d_%H%M%S"));
        let pipeline = WorldRecordPipeline::new(config, cuda_device)?;
        let mut pipeline = pipeline.with_telemetry(&run_id)?;
        pipeline.optimize_world_record(&graph, &kuramoto)?
    };

    let elapsed = start.elapsed();

    // Final Report
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                    FINAL RESULTS                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("📈 Results:");
    println!("   Chromatic Number: {} colors", result.chromatic_number);
    println!("   Conflicts: {}", result.conflicts);
    println!("   Quality Score: {:.4}", result.quality_score);
    println!("   Computation Time: {:.2}s", result.computation_time_ms / 1000.0);
    println!("   Total Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!();

    println!("🎯 World Record Comparison:");
    println!("   World Record: 83 colors");
    println!("   Our Result: {} colors", result.chromatic_number);

    if result.conflicts == 0 {
        let gap = result.chromatic_number as i32 - 83;
        if gap <= 0 {
            println!("   Status: 🏆 WORLD RECORD MATCHED/BEATEN!");
        } else {
            println!("   Gap: +{} colors ({:.1}%)", gap, (gap as f64 / 83.0) * 100.0);

            if result.chromatic_number <= 90 {
                println!("   Status: ✨ EXCELLENT (within 10% of WR)");
            } else if result.chromatic_number <= 100 {
                println!("   Status: ✅ STRONG (< 100 colors achieved)");
            } else {
                println!("   Status: 🔄 Room for improvement");
            }
        }
    } else {
        println!("   Status: ⚠️  Invalid coloring ({} conflicts)", result.conflicts);
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              WORLD RECORD ATTEMPT COMPLETE                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}

fn run_dsatur_solver(edges: Vec<(usize, usize)>, num_vertices: usize) -> Result<()> {
    // Build adjacency matrix
    println!("\n🔧 Building adjacency matrix...");
    let mut adjacency = Array2::from_elem((num_vertices, num_vertices), false);
    for (u, v) in edges.iter() {
        adjacency[[*u, *v]] = true;
        adjacency[[*v, *u]] = true;
    }

    // Create solver with max_colors from config (82 for world record)
    let max_colors = 82;
    println!("\n🔧 Initializing DSATUR solver...");
    println!("   Max colors (upper bound): {}", max_colors);

    let mut solver = DSaturSolver::new(adjacency.clone(), max_colors);

    // Optional: Create a warm start solution (simulated here)
    // In practice, this would come from thermodynamic sampling
    let warm_start = create_warm_start_solution(num_vertices);

    // MAIN SOLVE
    println!("\n{}", "=".repeat(70));
    println!("🚀 STARTING WORLD RECORD ATTEMPT");
    println!("{}", "=".repeat(70));

    let start_time = Instant::now();

    let result = solver.find_coloring(warm_start)?;

    let elapsed = start_time.elapsed();

    // Validate and analyze
    let chromatic = result.chromatic_number;
    let is_valid = result.is_valid(&adjacency);
    let conflicts = count_conflicts(&result.colors, &adjacency);

    println!("\n{}", "=".repeat(70));
    println!("📊 FINAL RESULTS");
    println!("{}", "=".repeat(70));

    println!("\n   Chromatic number: {} colors", chromatic);
    println!("   Conflicts: {}", conflicts);
    println!("   Time: {:.2} seconds", elapsed.as_secs_f64());
    println!("   Valid: {}", if is_valid { "✅ YES" } else { "❌ NO" });

    // World record check
    if chromatic <= 82 && is_valid {
        println!("\n{}", "🏆".repeat(35));
        println!("\n   🎉 WORLD RECORD ACHIEVED! 🎉");
        println!("   DSJC1000.5 colored with {} colors!", chromatic);
        println!("\n{}", "🏆".repeat(35));

        save_solution(&result.colors, chromatic)?;
    } else if chromatic <= 85 && is_valid {
        println!("\n⭐ EXCELLENT RESULT!");
        println!("   Only {} colors above world record", chromatic - 82);
    } else if chromatic <= 90 && is_valid {
        println!("\n✨ Very Good Result");
        println!("   {} colors (world record is 82)", chromatic);
    } else {
        println!("\n📈 Result Analysis:");
        if chromatic > 82 {
            println!("   Gap to world record: {} colors", chromatic - 82);
        }
        if !is_valid {
            println!("   ⚠️  Solution has conflicts - needs repair");
        }
    }

    println!("\n✅ Run complete!");
    Ok(())
}

/// Create a warm start solution (placeholder - would come from thermodynamic sampling)
fn create_warm_start_solution(num_vertices: usize) -> Option<ColoringSolution> {
    // Simulate a warm start with 115 colors (above the max_colors of 82)
    // This tests the warm start adjustment feature
    println!("\n🔥 Creating warm start solution...");
    println!("   Simulating thermodynamic pre-sampling with 115 colors");

    let colors: Vec<usize> = (0..num_vertices).map(|i| i % 115).collect();
    Some(ColoringSolution::new(colors))
}

fn count_conflicts(coloring: &[usize], adjacency: &Array2<bool>) -> usize {
    let n = adjacency.nrows();
    let mut conflicts = 0;

    for i in 0..n {
        for j in i + 1..n {
            if adjacency[[i, j]] && coloring[i] == coloring[j] {
                conflicts += 1;
            }
        }
    }

    conflicts
}

fn save_solution(coloring: &[usize], chromatic: usize) -> Result<()> {
    use std::fs;

    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let filename = format!(
        "world_record_DSJC1000.5_{}colors_{}.txt",
        chromatic, timestamp
    );

    let mut content = format!("# DSJC1000.5 Solution\n");
    content.push_str(&format!("# Chromatic Number: {}\n", chromatic));
    content.push_str(&format!("# Timestamp: {}\n", timestamp));
    content.push_str(&format!("# Solver: PRISM-AI DSATUR with Warm Start\n\n"));

    for (i, &color) in coloring.iter().enumerate() {
        content.push_str(&format!("{} {}\n", i + 1, color + 1));
    }

    fs::write(&filename, content)?;
    println!("\n💾 Solution saved to: {}", filename);

    Ok(())
}
