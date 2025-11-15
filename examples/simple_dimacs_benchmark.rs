///! Simple DIMACS Benchmark Runner for PRISM-AI
///!
///! Tests the PRISM platform against standard DIMACS graph coloring benchmarks
use prism_ai::{data::DIMACParser, PrismAI, PrismConfig};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== PRISM-AI DIMACS Benchmark Runner ===\n");

    let args: Vec<String> = std::env::args().collect();

    // Get benchmark directory from command line or use default
    let benchmark_dir = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/home/diddy/Downloads/PRISM-master/benchmarks/dimacs".to_string());

    // Get number of attempts from command line (default: 1000)
    let num_attempts = if args.len() > 2 {
        args[2].parse::<usize>().unwrap_or_else(|_| {
            eprintln!(
                "Warning: Invalid attempts value '{}', using default 1000",
                args[2]
            );
            1000
        })
    } else {
        1000
    };

    println!("Benchmark directory: {}", benchmark_dir);
    println!("Number of attempts: {}\n", num_attempts);

    // Best known chromatic numbers from literature
    let best_known = vec![
        ("DSJC125.1", 5),
        ("DSJC125.5", 17),
        ("DSJC125.9", 44),
        ("DSJC250.5", 28),
        ("DSJC500.5", 48),
        ("DSJC1000.5", 82), // WORLD RECORD TARGET
        ("DSJR500.1", 12),
        ("queen8_8", 9),
        ("queen11_11", 11),
        ("myciel6", 7),
        ("le450_25a", 25),
    ];

    // Configure PRISM
    let mut config = PrismConfig::default();
    config.use_gpu = cfg!(feature = "cuda");
    config.max_iterations = 1000;
    config.num_replicas = num_attempts; // Use configurable attempts
    config.temperature = 1.5; // Increased for more exploration

    println!("Configuration:");
    println!("  GPU Acceleration: {}", config.use_gpu);
    println!("  Max Iterations: {}", config.max_iterations);
    println!("  Number of Replicas: {}", config.num_replicas);
    println!("  Temperature: {}", config.temperature);
    println!("  GNN Enabled: {}", config.use_gnn);
    println!();

    let prism = PrismAI::new(config)?;

    println!(
        "{:<15} {:>8} {:>8} {:>10} {:>8} {:>8} {:>8}",
        "Graph", "Vertices", "Edges", "Time (ms)", "Colors", "Best", "Gap %"
    );
    println!("{}", "=".repeat(75));

    let mut total_tests = 0;
    let mut successful_tests = 0;

    for (graph_name, best_chromatic) in &best_known {
        let file_path = Path::new(&benchmark_dir).join(format!("{}.col", graph_name));

        if !file_path.exists() {
            println!("{:<15} [SKIP - File not found]", graph_name);
            continue;
        }

        total_tests += 1;

        // Parse the DIMACS file
        let adjacency = match DIMACParser::parse_file(&file_path) {
            Ok(adj) => adj,
            Err(e) => {
                println!("{:<15} [ERROR: {}]", graph_name, e);
                continue;
            }
        };

        let num_vertices = adjacency.len();
        let num_edges = adjacency.iter().map(|row| row.len()).sum::<usize>() / 2;

        // Run the coloring algorithm
        let start = Instant::now();
        let colors = match prism.color_graph(adjacency) {
            Ok(c) => c,
            Err(e) => {
                println!(
                    "{:<15} {:>8} {:>8} [ERROR: {}]",
                    graph_name, num_vertices, num_edges, e
                );
                continue;
            }
        };
        let duration = start.elapsed();

        // Count unique colors used
        let num_colors = colors.iter().max().unwrap_or(&0) + 1;

        // Calculate gap to best known
        let gap_percent = if *best_chromatic > 0 {
            ((num_colors as f64 - *best_chromatic as f64) / *best_chromatic as f64) * 100.0
        } else {
            0.0
        };

        successful_tests += 1;

        println!(
            "{:<15} {:>8} {:>8} {:>10.2} {:>8} {:>8} {:>7.1}%",
            graph_name,
            num_vertices,
            num_edges,
            duration.as_secs_f64() * 1000.0,
            num_colors,
            best_chromatic,
            gap_percent
        );
    }

    println!("{}", "=".repeat(75));
    println!("\nSummary:");
    println!("  Total Benchmarks: {}", total_tests);
    println!("  Successful Runs: {}", successful_tests);
    println!(
        "  Success Rate: {:.1}%",
        (successful_tests as f64 / total_tests as f64) * 100.0
    );

    Ok(())
}
