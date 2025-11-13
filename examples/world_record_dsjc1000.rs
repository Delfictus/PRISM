///! World Record Breaking Attempt on DSJC1000.5
///!
///! Uses the complete PRISM WorldRecordPipeline with:
///! - GPU-accelerated neuromorphic reservoir computing
///! - Active Inference policy selection
///! - ADP Q-learning parameter tuning
///! - Thermodynamic equilibration
///! - Quantum-Classical hybrid with feedback
///! - Memetic algorithm with TSP guidance
///! - Ensemble consensus voting
///! - Adaptive loopback for stagnation
///!
///! Target: 83 colors (world record)
///! Current best: 115 colors
///! Gap: 32 colors (27.8%)
use anyhow::Result;
use prism_ai::data::DimacsGraph;
use std::path::Path;
use std::sync::Arc;
use std::thread;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

#[cfg(feature = "cuda")]
use prct_core::world_record_pipeline::{WorldRecordConfig, WorldRecordPipeline};

#[cfg(feature = "cuda")]
use prct_core::gpu::{device_topology, device_profile::DeviceProfile};

#[cfg(feature = "cuda")]
use shared_types::KuramotoState;

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║   WORLD RECORD ATTEMPT: DSJC1000.5                        ║");
    println!("║   PRISM Ultimate Pipeline - Full GPU Acceleration         ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Load DSJC1000.5
    let graph_path = Path::new("benchmarks/dimacs/DSJC1000.5.col");

    println!("📊 Loading DSJC1000.5...");
    let dimacs = DimacsGraph::from_file(graph_path).map_err(|e| anyhow::anyhow!(e))?;

    // Convert adjacency matrix to edge list
    let mut edges = Vec::new();
    for i in 0..dimacs.num_vertices {
        for j in (i + 1)..dimacs.num_vertices {
            if dimacs.adjacency[[i, j]] {
                edges.push((i, j, 1.0));
            }
        }
    }

    // Flatten adjacency matrix to Vec<bool> (row-major order)
    let mut adjacency_flat = Vec::with_capacity(dimacs.num_vertices * dimacs.num_vertices);
    for i in 0..dimacs.num_vertices {
        for j in 0..dimacs.num_vertices {
            adjacency_flat.push(dimacs.adjacency[[i, j]]);
        }
    }

    // Convert to Graph
    let graph = shared_types::Graph {
        num_vertices: dimacs.num_vertices,
        num_edges: edges.len(),
        edges,
        adjacency: adjacency_flat,
        coordinates: None,
    };

    println!("✅ Graph loaded:");
    println!("   Vertices: {}", graph.num_vertices);
    println!("   Edges: {}", graph.num_edges);
    println!(
        "   Density: {:.1}%",
        dimacs.characteristics.edge_density * 100.0
    );
    println!("   Best known: 83 colors (world record)");
    println!();

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

    println!(
        "✅ Kuramoto initialized with {} oscillators",
        graph.num_vertices
    );
    println!();

    // Load configuration from file (with command-line override)
    let cfg_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "foundation/prct-core/configs/world_record.v1.toml".to_string());

    println!("📂 Loading configuration from: {}", cfg_path);
    let config = WorldRecordConfig::from_file(&cfg_path)?;
    println!("✅ Configuration loaded and validated");
    println!();

    println!("🎯 Pipeline Configuration:");
    println!("   Target: {} colors", config.target_chromatic);
    println!("   Max Runtime: {:.1} hours", config.max_runtime_hours);
    println!("   Workers: {}", config.num_workers);
    println!(
        "   GPU Reservoir: {}",
        if config.use_reservoir_prediction {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   Active Inference: {}",
        if config.use_active_inference {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   ADP Q-Learning: {}",
        if config.use_adp_learning {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   Thermodynamic: {}",
        if config.use_thermodynamic_equilibration {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   Quantum-Classical: {}",
        if config.use_quantum_classical_hybrid {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   Multi-Scale: {}",
        if config.use_multiscale_analysis {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "   Ensemble Consensus: {}",
        if config.use_ensemble_consensus {
            "✅"
        } else {
            "❌"
        }
    );
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
    println!(
        "   Computation Time: {:.2}s",
        result.computation_time_ms / 1000.0
    );
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
            println!(
                "   Gap: +{} colors ({:.1}%)",
                gap,
                (gap as f64 / 83.0) * 100.0
            );

            if result.chromatic_number <= 90 {
                println!("   Status: ✨ EXCELLENT (within 10% of WR)");
            } else if result.chromatic_number <= 100 {
                println!("   Status: ✅ STRONG (< 100 colors achieved)");
            } else {
                println!("   Status: 🔄 Room for improvement");
            }
        }
    } else {
        println!(
            "   Status: ⚠️  Invalid coloring ({} conflicts)",
            result.conflicts
        );
    }

    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              WORLD RECORD ATTEMPT COMPLETE                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() -> Result<()> {
    println!("❌ This benchmark requires CUDA support.");
    println!(
        "   Rebuild with: cargo run --release --features cuda --example world_record_dsjc1000"
    );
    Ok(())
}
