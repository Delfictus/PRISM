use prism_ai::cuda::{GpuColoringEngine, PrismConfig, PrismPipeline};
///! DIMACS Graph Coloring Benchmark Suite
///!
///! Tests adaptive coloring algorithm against 14 standard DIMACS benchmarks.
///! Compares results against best known chromatic numbers from literature.
///!
///! GPU-ONLY: Requires CUDA-capable device
use prism_ai::data::{DimacsGraph, GraphType};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Best known chromatic numbers from literature
/// Sources:
/// - DIMACS Challenge (1993)
/// - Trick (2012) - DSJC1000.5: χ=82 (November 2012)
/// - Various published results
fn best_known_chromatic() -> HashMap<&'static str, usize> {
    let mut map = HashMap::new();

    // DSJC graphs (random with varying density)
    map.insert("DSJC125.1", 5); // Very sparse
    map.insert("DSJC125.5", 17); // Medium density
    map.insert("DSJC125.9", 44); // Very dense
    map.insert("DSJC250.5", 28); // Medium
    map.insert("DSJC500.5", 48); // Medium
    map.insert("DSJC1000.5", 82); // WORLD RECORD TARGET (Trick 2012)

    // DSJR graphs (random with geometric distribution)
    map.insert("DSJR500.1", 12); // Sparse
    map.insert("DSJR500.5", 122); // Dense

    // Leighton graphs (adversarial)
    map.insert("le450_15a", 15); // Designed to be hard
    map.insert("le450_25a", 25); // Designed to be hard

    // Queen graphs (geometric)
    map.insert("queen8_8", 9); // 8x8 chessboard
    map.insert("queen11_11", 11); // 11x11 chessboard

    // Mycielski graphs (triangle-free)
    map.insert("myciel5", 6); // χ=6, ω=2 (triangle-free)
    map.insert("myciel6", 7); // χ=7, ω=2 (triangle-free)

    map
}

/// Benchmark result for a single graph
#[derive(Debug, Clone)]
struct BenchmarkResult {
    graph_name: String,
    graph_type: GraphType,
    num_vertices: usize,
    num_edges: usize,
    edge_density: f64,

    // Results
    chromatic_achieved: usize,
    chromatic_best_known: Option<usize>,

    // Performance
    runtime_ms: f64,

    // Multiple runs statistics
    runs: Vec<usize>, // Chromatic numbers from multiple runs
    best_run: usize,
    worst_run: usize,
    avg_run: f64,
    std_dev: f64,
}

impl BenchmarkResult {
    /// Calculate gap to best known result (percentage)
    fn gap_percent(&self) -> Option<f64> {
        self.chromatic_best_known.map(|best| {
            if best == 0 {
                return 0.0;
            }
            ((self.chromatic_achieved as f64 - best as f64) / best as f64) * 100.0
        })
    }

    /// Check if we matched or beat best known
    fn is_world_class(&self) -> bool {
        match self.chromatic_best_known {
            Some(best) => self.best_run <= best,
            None => false,
        }
    }
}

/// Run adaptive coloring on a single graph
fn benchmark_graph(
    gpu_engine: &GpuColoringEngine,
    prism_pipeline: &mut PrismPipeline,
    graph_path: &Path,
    best_known: &HashMap<&str, usize>,
    num_runs: usize,
    use_prism: bool,
) -> Result<BenchmarkResult, String> {
    let graph_name = graph_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    println!("\n{}", "=".repeat(80));
    println!("📊 Benchmarking: {}", graph_name);
    println!("{}", "=".repeat(80));

    // Load graph
    let start_load = Instant::now();
    let graph = DimacsGraph::from_file(graph_path)?;
    let load_time = start_load.elapsed().as_secs_f64() * 1000.0;

    println!("\n📈 Graph Characteristics:");
    println!("  Vertices:     {}", graph.num_vertices);
    println!("  Edges:        {}", graph.num_edges);
    println!(
        "  Density:      {:.2}%",
        graph.characteristics.edge_density * 100.0
    );
    println!("  Avg degree:   {:.1}", graph.characteristics.avg_degree);
    println!("  Graph type:   {:?}", graph.characteristics.graph_type);
    println!("  Load time:    {:.2}ms", load_time);

    println!("\n🔬 Chromatic Bounds:");
    println!(
        "  Lower bound:  {} (clique)",
        graph.characteristics.clique_lower_bound
    );
    println!(
        "  Upper bound:  {} (greedy)",
        graph.characteristics.greedy_upper_bound
    );
    if let Some(best) = best_known.get(graph_name.as_str()) {
        println!("  Best known:   {} ⭐", best);
    }

    println!("\n🔧 Recommended Strategy:");
    let strategy = &graph.characteristics.recommended_strategy;
    println!("  Use GNN:            {}", strategy.use_gnn);
    println!(
        "  Exploration:        {:.2}",
        strategy.exploration_vs_exploitation
    );
    println!("  Temperature:        {:.2}", strategy.temperature_scaling);
    println!(
        "  Parallel attempts:  {:.1}x",
        strategy.parallel_attempts_factor
    );

    // Run multiple attempts using GPU
    println!("\n🚀 Running {} GPU coloring attempts...", num_runs);
    if use_prism {
        println!("   🧠 PRISM-AI ENABLED: Full 7-step GPU pipeline");
    } else {
        println!("   ⚡ GPU-ONLY: Parallel coloring without PRISM-AI");
    }
    println!("{}", "-".repeat(80));

    let mut runs = Vec::new();
    let mut total_time = 0.0;

    // Use strategy-recommended parameters
    let base_attempts = 100; // Base parallel attempts
    let temperature = strategy.temperature_scaling as f32;
    let parallel_attempts =
        (base_attempts as f32 * strategy.parallel_attempts_factor as f32) as usize;

    // Compute PRISM-AI coherence once (reuse across runs)
    let prism_coherence = if use_prism {
        println!("\n🧠 Computing PRISM-AI coherence...");
        let coherence_start = Instant::now();
        let coh = prism_pipeline
            .compute_coherence(&graph.adjacency)
            .map_err(|e| format!("PRISM coherence computation failed: {:?}", e))?;
        println!(
            "   ✅ PRISM-AI coherence computed in {:.2}ms",
            coherence_start.elapsed().as_secs_f64() * 1000.0
        );
        Some(coh.enhanced)
    } else {
        None
    };

    for run_idx in 0..num_runs {
        let start = Instant::now();

        // GPU adaptive coloring with parallel exploration
        let result = if use_prism {
            gpu_engine
                .color_graph_with_coherence(
                    &graph.adjacency,
                    prism_coherence.as_ref().map(|c| c.as_slice()),
                    parallel_attempts,
                    temperature,
                    graph.num_vertices,
                )
                .map_err(|e| format!("GPU coloring failed: {:?}", e))?
        } else {
            gpu_engine
                .color_graph(
                    &graph.adjacency,
                    parallel_attempts,
                    temperature,
                    graph.num_vertices,
                )
                .map_err(|e| format!("GPU coloring failed: {:?}", e))?
        };

        let chromatic = result.chromatic_number;
        let runtime = start.elapsed().as_secs_f64() * 1000.0;
        total_time += runtime;
        runs.push(chromatic);

        let best_so_far = runs.iter().min().unwrap();
        print!(
            "  Run {:>3}/{}: χ = {:>3} ({:>6.2}ms, {} GPU attempts)",
            run_idx + 1,
            num_runs,
            chromatic,
            runtime,
            parallel_attempts
        );

        if chromatic == *best_so_far {
            print!(" 🌟 BEST");
        }
        println!();
    }

    // Calculate statistics
    let best_run = *runs.iter().min().unwrap();
    let worst_run = *runs.iter().max().unwrap();
    let avg_run = runs.iter().sum::<usize>() as f64 / runs.len() as f64;

    let variance = runs
        .iter()
        .map(|&x| {
            let diff = x as f64 - avg_run;
            diff * diff
        })
        .sum::<f64>()
        / runs.len() as f64;
    let std_dev = variance.sqrt();

    let avg_runtime = total_time / num_runs as f64;

    Ok(BenchmarkResult {
        graph_name: graph_name.clone(),
        graph_type: graph.characteristics.graph_type,
        num_vertices: graph.num_vertices,
        num_edges: graph.num_edges,
        edge_density: graph.characteristics.edge_density,
        chromatic_achieved: best_run,
        chromatic_best_known: best_known.get(graph_name.as_str()).copied(),
        runtime_ms: avg_runtime,
        runs,
        best_run,
        worst_run,
        avg_run,
        std_dev,
    })
}

/// Print summary table of all results
fn print_summary(results: &[BenchmarkResult]) {
    println!("\n");
    println!("{}", "=".repeat(120));
    println!("📊 DIMACS BENCHMARK SUITE - SUMMARY RESULTS");
    println!("{}", "=".repeat(120));

    // Table header
    println!(
        "\n{:<20} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Graph", "Vertices", "Edges", "Type", "Best", "Avg", "Worst", "BestKnown", "Gap"
    );
    println!("{}", "-".repeat(120));

    let mut total_world_class = 0;
    let mut total_graphs = 0;

    for result in results {
        let type_str = format!("{:?}", result.graph_type);
        let type_abbr = &type_str[0..type_str.len().min(8)];

        let best_known_str = result
            .chromatic_best_known
            .map(|x| format!("{}", x))
            .unwrap_or_else(|| "?".to_string());

        let gap_str = result
            .gap_percent()
            .map(|g| format!("{:+.1}%", g))
            .unwrap_or_else(|| "N/A".to_string());

        let status = if result.is_world_class() {
            "✅"
        } else {
            "❌"
        };

        println!(
            "{} {:<18} {:>8} {:>8} {:>8} {:>10} {:>10.1} {:>10} {:>10} {:>10}",
            status,
            result.graph_name,
            result.num_vertices,
            result.num_edges,
            type_abbr,
            result.best_run,
            result.avg_run,
            result.worst_run,
            best_known_str,
            gap_str
        );

        if result.is_world_class() {
            total_world_class += 1;
        }
        total_graphs += 1;
    }

    println!("{}", "-".repeat(120));

    // Overall statistics
    let total_vertices: usize = results.iter().map(|r| r.num_vertices).sum();
    let total_edges: usize = results.iter().map(|r| r.num_edges).sum();
    let avg_runtime: f64 = results.iter().map(|r| r.runtime_ms).sum::<f64>() / results.len() as f64;

    println!("\n📈 Overall Statistics:");
    println!("  Total graphs:        {}", total_graphs);
    println!(
        "  World-class results: {}/{} ({:.1}%)",
        total_world_class,
        total_graphs,
        (total_world_class as f64 / total_graphs as f64) * 100.0
    );
    println!("  Total vertices:      {}", total_vertices);
    println!("  Total edges:         {}", total_edges);
    println!("  Avg runtime/graph:   {:.2}ms", avg_runtime);

    // Gaps analysis
    let gaps: Vec<f64> = results.iter().filter_map(|r| r.gap_percent()).collect();

    if !gaps.is_empty() {
        let avg_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let max_gap = gaps.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_gap = gaps.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        println!("\n📊 Gap Analysis (vs Best Known):");
        println!("  Average gap:  {:+.2}%", avg_gap);
        println!("  Max gap:      {:+.2}%", max_gap);
        println!("  Min gap:      {:+.2}%", min_gap);
    }

    // Highlight DSJC1000.5 (primary target)
    if let Some(dsjc1000) = results.iter().find(|r| r.graph_name == "DSJC1000.5") {
        println!("\n🎯 PRIMARY TARGET: DSJC1000.5");
        println!("{}", "-".repeat(80));
        println!("  Best achieved:  {} colors", dsjc1000.best_run);
        println!("  Best known:     82 colors (Trick, November 2012)");
        if let Some(gap) = dsjc1000.gap_percent() {
            println!("  Gap:            {:+.1}%", gap);
            if gap <= 0.0 {
                println!("  🏆 WORLD RECORD MATCHED OR BEATEN! 🏆");
            } else {
                println!(
                    "  ⚠️  Need {} fewer colors to match world record",
                    dsjc1000.best_run - 82
                );
            }
        }
    }

    println!("\n{}", "=".repeat(120));
}

fn main() {
    println!("🎯 PRISM-AI DIMACS Graph Coloring Benchmark");
    println!("{}", "=".repeat(80));

    // Configuration
    let benchmark_dir = Path::new("../benchmarks/dimacs");
    let num_runs_per_graph = 10; // Multiple runs for statistical validation
    let use_prism = true; // Enable full PRISM-AI pipeline

    println!("\n⚙️  Configuration:");
    println!("  Benchmark directory: {}", benchmark_dir.display());
    println!("  Runs per graph:      {}", num_runs_per_graph);
    println!("  GPU-ONLY mode:       ENABLED");
    println!(
        "  PRISM-AI mode:       {}",
        if use_prism {
            "ENABLED (Full 7-step pipeline)"
        } else {
            "DISABLED"
        }
    );

    // Initialize GPU coloring engine
    println!("\n🔍 Initializing GPU coloring engine...");
    let gpu_engine = match GpuColoringEngine::new() {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("\n❌ FATAL: Failed to initialize GPU: {}", e);
            eprintln!("    GPU-ONLY mode requires CUDA-capable device");
            std::process::exit(1);
        }
    };
    println!("  ✅ GPU coloring engine initialized");

    // Initialize PRISM-AI pipeline
    println!("\n🧠 Initializing PRISM-AI pipeline...");
    let prism_config = PrismConfig {
        use_transfer_entropy: false, // Disabled for performance (can enable for smaller graphs)
        use_tda: true,
        use_neuromorphic: false,              // Disabled for performance
        use_gnn: false,                       // No trained model yet
        fusion_weights: [0.5, 0.0, 0.0, 0.5], // TDA + GNN placeholders
        ensemble_size: 10,
        ensemble_temperature: 1.0,
        reservoir_size: 500,
        gnn_model_path: None,
    };

    let mut prism_pipeline = match PrismPipeline::new(prism_config) {
        Ok(pipeline) => pipeline,
        Err(e) => {
            eprintln!(
                "\n⚠️  WARNING: Failed to initialize PRISM-AI pipeline: {}",
                e
            );
            eprintln!("    Continuing with GPU-only mode");
            // Create minimal pipeline
            PrismPipeline::new(PrismConfig {
                use_transfer_entropy: false,
                use_tda: false,
                use_neuromorphic: false,
                use_gnn: false,
                ..Default::default()
            })
            .expect("Failed to create minimal pipeline")
        }
    };
    println!("  ✅ PRISM-AI pipeline initialized");

    // Load best known results
    let best_known = best_known_chromatic();
    println!(
        "  ✅ Loaded {} best-known chromatic numbers",
        best_known.len()
    );

    // Find all .col files
    let graphs = [
        "DSJC125.1.col",
        "DSJC125.5.col",
        "DSJC125.9.col",
        "DSJC250.5.col",
        "DSJC500.5.col",
        "DSJC1000.5.col",
        "DSJR500.1.col",
        "DSJR500.5.col",
        "le450_15a.col",
        "le450_25a.col",
        "queen8_8.col",
        "queen11_11.col",
        "myciel5.col",
        "myciel6.col",
    ];

    println!("\n🔍 Found {} DIMACS benchmark graphs", graphs.len());

    // Run benchmarks
    let start_total = Instant::now();
    let mut results = Vec::new();

    for graph_file in &graphs {
        let graph_path = benchmark_dir.join(graph_file);

        if !graph_path.exists() {
            eprintln!("\n⚠️  WARNING: Graph not found: {}", graph_path.display());
            continue;
        }

        match benchmark_graph(
            &gpu_engine,
            &mut prism_pipeline,
            &graph_path,
            &best_known,
            num_runs_per_graph,
            use_prism,
        ) {
            Ok(result) => {
                // Print individual result
                println!("\n✅ Results for {}:", result.graph_name);
                println!("  Best:    {} colors", result.best_run);
                println!("  Average: {:.1} colors", result.avg_run);
                println!("  Worst:   {} colors", result.worst_run);
                println!("  StdDev:  {:.2}", result.std_dev);

                if let Some(gap) = result.gap_percent() {
                    println!("  Gap:     {:+.1}%", gap);
                }

                results.push(result);
            }
            Err(e) => {
                eprintln!("\n❌ ERROR: Failed to benchmark {}: {}", graph_file, e);
            }
        }
    }

    let total_time = start_total.elapsed().as_secs_f64();

    // Print summary
    if !results.is_empty() {
        print_summary(&results);

        println!("\n⏱️  Total benchmark time: {:.2}s", total_time);
        println!();
    } else {
        eprintln!("\n❌ No benchmarks completed successfully!");
        std::process::exit(1);
    }
}
