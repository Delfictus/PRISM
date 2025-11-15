//! Universal PRISM CLI - Accepts all types of data files
//!
//! Supports:
//! - MTX files (Matrix Market format)
//! - Protein data (PDB, CIF, MTX protein graphs)
//! - Graph data (DIMACS, MTX, edge lists)
//! - CSV/TSV data
//! - Custom matrix formats

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "prism-universal")]
#[command(about = "Universal PRISM platform for all data types", long_about = None)]
struct Args {
    /// Input file path (supports MTX, PDB, CIF, DIMACS, CSV formats)
    #[arg(short, long)]
    input: PathBuf,

    /// Number of optimization attempts
    #[arg(short, long, default_value = "1000")]
    attempts: usize,

    /// Output directory for results
    #[arg(short, long, default_value = "./output")]
    output: PathBuf,

    /// Input file type (auto-detect if not specified)
    #[arg(short = 't', long)]
    file_type: Option<String>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Use GPU acceleration (default: true)
    #[arg(long, default_value = "true")]
    gpu: bool,

    /// Algorithm to use: greedy, prct (default: greedy)
    #[arg(short = 'a', long, default_value = "greedy")]
    algorithm: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          PRISM Universal Platform - GPU Accelerated         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Validate input file exists
    if !args.input.exists() {
        anyhow::bail!("Input file not found: {}", args.input.display());
    }

    // Detect file type
    let file_type = detect_file_type(&args.input, args.file_type.clone())?;

    println!("📂 Input file: {}", args.input.display());
    println!("📊 File type: {}", file_type);
    println!("🎯 Attempts: {}", args.attempts);
    println!("🧮 Algorithm: {}", args.algorithm);
    println!("💾 Output: {}", args.output.display());
    println!("🚀 GPU: {}", if args.gpu { "Enabled" } else { "Disabled" });
    println!();

    // Create output directory
    std::fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    // Process based on file type
    match file_type.as_str() {
        "mtx" | "matrix_market" => process_mtx_file(&args)?,
        "pdb" => process_protein_file(&args)?,
        "dimacs" | "col" => process_dimacs_file(&args)?,
        "csv" | "tsv" => process_tabular_file(&args)?,
        _ => anyhow::bail!("Unsupported file type: {}", file_type),
    }

    println!();
    println!("✅ Processing complete!");
    println!("📁 Results saved to: {}", args.output.display());

    Ok(())
}

fn detect_file_type(path: &PathBuf, explicit_type: Option<String>) -> Result<String> {
    if let Some(ft) = explicit_type {
        return Ok(ft);
    }

    // Auto-detect from extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let file_type = match extension.as_str() {
        "mtx" => "mtx",
        "pdb" => "pdb",
        "cif" => "pdb", // Treat CIF as protein data
        "col" => "dimacs",
        "csv" => "csv",
        "tsv" => "tsv",
        "txt" => {
            // Try to detect DIMACS format by reading first line
            if let Ok(content) = std::fs::read_to_string(path) {
                if content.starts_with("p edge") || content.starts_with("c ") {
                    "dimacs"
                } else {
                    "txt"
                }
            } else {
                "txt"
            }
        }
        _ => "unknown",
    };

    if file_type == "unknown" {
        anyhow::bail!("Could not detect file type. Please specify with --file-type");
    }

    Ok(file_type.to_string())
}

fn process_mtx_file(args: &Args) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    println!("🔍 Parsing MTX file...");

    let file = File::open(&args.input)?;
    let reader = BufReader::new(file);

    let mut num_vertices = 0;
    let mut num_edges = 0;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip comments and empty lines
        if line.starts_with('%') || line.is_empty() {
            continue;
        }

        // Parse header line
        if num_vertices == 0 {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                num_vertices = parts[0].parse().unwrap_or(0);
                num_edges = parts[2].parse().unwrap_or(0);
                println!("  ✓ {} vertices, {} edges", num_vertices, num_edges);
                continue;
            }
        }

        // Parse edge
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            if let (Ok(u), Ok(v)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                edges.push((u - 1, v - 1)); // Convert to 0-indexed
            }
        }
    }

    println!("  ✓ Loaded {} edges", edges.len());
    println!();
    println!(
        "🚀 Running PRISM optimization with {} attempts...",
        args.attempts
    );

    // Run optimization with chosen algorithm
    run_optimization(
        num_vertices,
        &edges,
        args.attempts,
        args.gpu,
        &args.algorithm,
    )?;

    Ok(())
}

fn process_protein_file(args: &Args) -> Result<()> {
    use prism_ai::protein_parser::ProteinContactGraph;

    println!("🧬 Processing protein structure file...");
    println!("  📖 Parsing PDB format...");

    // Parse PDB file and build contact graph
    // Contact distance threshold: 8.0 Angstroms (typical for residue contacts)
    let contact_distance = 8.0;
    let contact_graph = ProteinContactGraph::from_pdb_file(&args.input, contact_distance)
        .context("Failed to parse PDB file")?;

    println!("  ✓ {}", contact_graph.summary());

    if contact_graph.num_edges == 0 {
        println!("  ⚠️  No residue contacts found");
        println!("  💡 Try increasing contact distance threshold");
        anyhow::bail!("Empty contact graph - cannot proceed with coloring");
    }

    println!();
    println!("🚀 Running PRISM optimization on protein contact graph...");

    // Extract graph data
    let num_vertices = contact_graph.num_vertices;
    let edges = contact_graph.get_edges();

    // Run graph coloring optimization
    run_optimization(
        num_vertices,
        &edges,
        args.attempts,
        args.gpu,
        &args.algorithm,
    )?;

    Ok(())
}

fn process_dimacs_file(args: &Args) -> Result<()> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    println!("🔍 Parsing DIMACS file...");

    let file = File::open(&args.input)?;
    let reader = BufReader::new(file);

    let mut num_vertices = 0;
    let mut num_edges = 0;
    let mut edges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip comments and empty lines
        if line.starts_with('c') || line.is_empty() {
            continue;
        }

        // Parse problem line
        if line.starts_with('p') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                num_vertices = parts[2].parse().unwrap_or(0);
                num_edges = parts[3].parse().unwrap_or(0);
                println!("  ✓ {} vertices, {} edges", num_vertices, num_edges);
                continue;
            }
        }

        // Parse edge
        if line.starts_with('e') {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                if let (Ok(u), Ok(v)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
                    edges.push((u - 1, v - 1)); // Convert to 0-indexed
                }
            }
        }
    }

    println!("  ✓ Loaded {} edges", edges.len());
    println!();
    println!(
        "🚀 Running PRISM optimization with {} attempts...",
        args.attempts
    );

    run_optimization(
        num_vertices,
        &edges,
        args.attempts,
        args.gpu,
        &args.algorithm,
    )?;

    Ok(())
}

fn process_tabular_file(args: &Args) -> Result<()> {
    println!("📊 Processing tabular data file...");
    println!("  ⚠️  Tabular data processing not yet implemented");
    println!("  💡 Future support for CSV/TSV adjacency matrices");

    Ok(())
}

fn run_optimization(
    num_vertices: usize,
    edges: &[(usize, usize)],
    attempts: usize,
    use_gpu: bool,
    algorithm: &str,
) -> Result<()> {
    use std::collections::HashMap;
    use std::time::Instant;

    let start = Instant::now();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  PRISM OPTIMIZATION IN PROGRESS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    if use_gpu {
        println!("🎮 GPU acceleration: ENABLED");
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
        {
            if let Ok(gpu_name) = String::from_utf8(output.stdout) {
                println!("  GPU: {}", gpu_name.trim());
            }
        }
    } else {
        println!("💻 Running on CPU");
    }

    println!();
    println!("  Graph: {} vertices, {} edges", num_vertices, edges.len());
    println!("  Optimization attempts: {}", attempts);
    println!("  Algorithm: {}", algorithm);
    println!();

    // === ACTUAL PRISM INTEGRATION ===

    // Phase 1: Build adjacency list from edges
    println!("  [1/5] Building graph representation...");
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); num_vertices];
    for &(u, v) in edges {
        if u < num_vertices && v < num_vertices {
            adjacency[u].push(v);
            adjacency[v].push(u); // Undirected graph
        }
    }
    println!("      ✓ Adjacency list created");

    // Phase 2: Initialize PRISM components
    println!("  [2/5] Initializing PRISM platform...");

    // Create ensemble generator for parallel search
    use prism_ai::cuda::EnsembleGenerator;
    let num_replicas = (attempts / 100).max(10).min(1000); // Smart replica count
    let temperature = 1.0;

    let ensemble_gen = EnsembleGenerator::new(num_replicas, temperature)
        .context("Failed to create ensemble generator")?;
    println!(
        "      ✓ Ensemble generator ready ({} replicas)",
        num_replicas
    );

    // Initialize coloring engine based on algorithm choice
    use prism_ai::cuda::gpu_coloring::GpuColoringEngine;
    use prism_ai::cuda::{PRCTAlgorithm, PRCTConfig};

    let use_prct = algorithm.to_lowercase() == "prct";

    let gpu_coloring = GpuColoringEngine::new()?;
    let prct_algorithm = if use_prct {
        let config = PRCTConfig {
            neuro_base_frequency: 20.0,
            neuro_time_window: 100.0,
            quantum_coupling_strength: 1.0,
            quantum_evolution_time: 1.0,
            kuramoto_coupling: 1.0,
            kuramoto_steps: 100,
            target_colors: None,
            coherence_threshold: 0.5,
            gpu_accelerated: use_gpu,
        };
        Some(PRCTAlgorithm::with_config(config).context("Failed to create PRCT algorithm")?)
    } else {
        None
    };

    if use_prct {
        println!("      ✓ PRCT algorithm initialized (GPU: {})", use_gpu);
    } else {
        println!("      ✓ Greedy coloring engine initialized");
    }

    // Phase 3: Generate diverse solution ensemble
    println!("  [3/5] Generating solution ensemble...");
    let orderings = ensemble_gen
        .generate(&adjacency)
        .context("Failed to generate ensemble")?;
    println!("      ✓ Generated {} diverse orderings", orderings.len());

    // Phase 4: Parallel coloring optimization
    println!(
        "  [4/5] Running {} coloring...",
        if use_prct { "PRCT" } else { "greedy" }
    );
    let mut best_coloring: Option<Vec<usize>> = None;
    let mut best_num_colors = usize::MAX;
    let mut valid_attempts = 0;

    // Progress tracking
    let progress_interval = attempts / 10;

    for (i, ordering) in orderings.iter().enumerate().take(attempts) {
        // Perform coloring with chosen algorithm
        let coloring_result = if let Some(ref prct) = prct_algorithm {
            prct.color(&adjacency, ordering)
        } else {
            // Greedy coloring not currently available - use PRCT instead
            anyhow::bail!("Greedy algorithm not implemented. Please use --algorithm prct")
        };

        match coloring_result {
            Ok(coloring) => {
                // Calculate number of colors used
                let num_colors = if coloring.is_empty() {
                    0
                } else {
                    *coloring.iter().max().unwrap_or(&0) + 1
                };

                // Track best solution
                if num_colors < best_num_colors && num_colors > 0 {
                    best_num_colors = num_colors;
                    best_coloring = Some(coloring.clone());
                    println!(
                        "      → New best: {} colors (attempt {}/{})",
                        best_num_colors,
                        i + 1,
                        orderings.len().min(attempts)
                    );
                }

                valid_attempts += 1;
            }
            Err(e) => {
                log::debug!("Coloring attempt {} failed: {}", i, e);
            }
        }

        // Progress indicator
        if i > 0 && i % progress_interval == 0 {
            let progress = (i as f64 / orderings.len().min(attempts) as f64) * 100.0;
            println!(
                "      Progress: {:.1}% ({}/{}) - Best: {} colors",
                progress,
                i,
                orderings.len().min(attempts),
                best_num_colors
            );
        }
    }

    println!("      ✓ Completed {} valid colorings", valid_attempts);

    // Phase 5: Validate and finalize
    println!("  [5/5] Validating best solution...");

    let (final_colors, is_valid) = if let Some(ref coloring) = best_coloring {
        // Validate coloring
        let mut valid = true;
        for (u, neighbors) in adjacency.iter().enumerate() {
            for &v in neighbors {
                if u < coloring.len() && v < coloring.len() {
                    if coloring[u] == coloring[v] {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                break;
            }
        }
        (best_num_colors, valid)
    } else {
        // Fallback: greedy coloring
        println!("      ⚠️  No GPU solution found, using greedy fallback...");
        let mut coloring = vec![0; num_vertices];
        let mut max_color = 0;

        for v in 0..num_vertices {
            let mut used_colors = vec![false; num_vertices];
            for &neighbor in &adjacency[v] {
                if neighbor < coloring.len() {
                    used_colors[coloring[neighbor]] = true;
                }
            }

            for color in 0..num_vertices {
                if !used_colors[color] {
                    coloring[v] = color;
                    max_color = max_color.max(color);
                    break;
                }
            }
        }

        best_coloring = Some(coloring);
        (max_color + 1, true)
    };

    println!(
        "      ✓ Solution validated: {}",
        if is_valid { "VALID ✓" } else { "INVALID ✗" }
    );

    let duration = start.elapsed();

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  RESULTS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  🎯 Best coloring: {} colors", final_colors);
    println!(
        "  ✅ Status: {}",
        if is_valid {
            "Valid"
        } else {
            "Invalid (conflicts detected)"
        }
    );
    println!("  🔍 Attempts evaluated: {}/{}", valid_attempts, attempts);
    println!("  ⏱️  Total time: {:.2}s", duration.as_secs_f64());
    println!(
        "  🚀 Throughput: {:.0} attempts/sec",
        valid_attempts as f64 / duration.as_secs_f64()
    );
    println!();

    // Save results if we have a valid coloring
    if let Some(coloring) = best_coloring {
        println!("  💾 Saving results...");
        save_coloring_results(
            num_vertices,
            edges.len(),
            final_colors,
            &coloring,
            duration.as_secs_f64(),
        )?;
        println!("      ✓ Results saved to output/coloring_result.json");
    }

    println!();

    Ok(())
}

fn save_coloring_results(
    num_vertices: usize,
    num_edges: usize,
    num_colors: usize,
    coloring: &[usize],
    time_seconds: f64,
) -> Result<()> {
    use std::fs;

    let results = serde_json::json!({
        "graph": {
            "vertices": num_vertices,
            "edges": num_edges,
        },
        "solution": {
            "num_colors": num_colors,
            "coloring": coloring,
        },
        "performance": {
            "time_seconds": time_seconds,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        }
    });

    fs::create_dir_all("output")?;
    let result_str = serde_json::to_string_pretty(&results)?;
    fs::write("output/coloring_result.json", result_str)?;

    Ok(())
}
