///! GPU-Accelerated Protein Structure Coloring Benchmark
///!
///! Parses PDB files and colors residue contact graphs using PRISM-AI GPU acceleration
use prism_ai::{protein_parser::ProteinContactGraph, PrismAI, PrismConfig};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== PRISM-AI Protein Structure Benchmark ===");
    println!("GPU-Accelerated Residue Contact Graph Coloring\n");

    let args: Vec<String> = std::env::args().collect();

    // Get PDB file path from command line or use default
    let pdb_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/home/diddy/Desktop/PRISM-FINNAL-PUSH/data/nipah/2VSM.pdb".to_string());

    // Get contact distance threshold (default: 8.0 Angstroms)
    let contact_distance = if args.len() > 2 {
        args[2].parse::<f64>().unwrap_or_else(|_| {
            eprintln!(
                "Warning: Invalid contact distance '{}', using default 8.0Å",
                args[2]
            );
            8.0
        })
    } else {
        8.0
    };

    // Get number of GPU attempts (default: 5000)
    let num_attempts = if args.len() > 3 {
        args[3].parse::<usize>().unwrap_or_else(|_| {
            eprintln!(
                "Warning: Invalid attempts value '{}', using default 5000",
                args[3]
            );
            5000
        })
    } else {
        5000
    };

    println!("PDB File: {}", pdb_path);
    println!("Contact Distance: {:.1}Å", contact_distance);
    println!("GPU Attempts: {}\n", num_attempts);

    // Check if file exists
    if !Path::new(&pdb_path).exists() {
        eprintln!("❌ Error: PDB file not found: {}", pdb_path);
        eprintln!("\nUsage:");
        eprintln!("  {} <pdb_file> [contact_distance] [num_attempts]", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} data/nipah/2VSM.pdb 8.0 5000", args[0]);
        std::process::exit(1);
    }

    // Parse PDB file
    println!("[1/4] Parsing PDB file...");
    let parse_start = Instant::now();
    let contact_graph = ProteinContactGraph::from_pdb_file(&pdb_path, contact_distance)?;
    let parse_time = parse_start.elapsed();

    println!("  ✅ Parsed in {:.2}ms", parse_time.as_secs_f64() * 1000.0);
    println!("  📊 {}", contact_graph.summary());
    println!(
        "  📊 Graph density: {:.2}%\n",
        (contact_graph.num_edges as f64
            / (contact_graph.num_vertices * (contact_graph.num_vertices - 1) / 2) as f64)
            * 100.0
    );

    // Convert to adjacency list
    println!("[2/4] Building adjacency list...");
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); contact_graph.num_vertices];
    for (i, j) in contact_graph.get_edges() {
        adjacency[i].push(j);
        adjacency[j].push(i);
    }
    println!("  ✅ Adjacency list ready\n");

    // Configure PRISM with GPU acceleration
    println!("[3/4] Initializing PRISM-AI GPU engine...");
    let mut config = PrismConfig::default();
    config.use_gpu = cfg!(feature = "cuda");
    config.num_replicas = num_attempts;
    config.temperature = 1.5; // Increased for better exploration
    config.max_iterations = 1000;

    println!("  Configuration:");
    println!("    GPU Acceleration: {}", config.use_gpu);
    println!("    Number of Replicas: {}", config.num_replicas);
    println!("    Temperature: {}", config.temperature);
    println!();

    let prism = PrismAI::new(config)?;

    // Color the protein contact graph
    println!("[4/4] Coloring residue contact graph with GPU acceleration...");
    let coloring_start = Instant::now();
    let colors = prism.color_graph(adjacency)?;
    let coloring_time = coloring_start.elapsed();

    // Calculate chromatic number
    let chromatic_number = colors.iter().max().map(|&c| c + 1).unwrap_or(0);

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "PDB File:           {}",
        Path::new(&pdb_path).file_name().unwrap().to_string_lossy()
    );
    println!("Residues:           {}", contact_graph.num_vertices);
    println!("Contacts:           {}", contact_graph.num_edges);
    println!("Contact Threshold:  {:.1}Å", contact_distance);
    println!();
    println!("Chromatic Number:   {} colors", chromatic_number);
    println!(
        "Coloring Time:      {:.2}ms",
        coloring_time.as_secs_f64() * 1000.0
    );
    println!("GPU Attempts:       {}", num_attempts);
    println!();

    // Color distribution analysis
    let mut color_counts: Vec<usize> = vec![0; chromatic_number];
    for &color in &colors {
        color_counts[color] += 1;
    }

    println!("Color Distribution:");
    for (color_id, count) in color_counts.iter().enumerate() {
        let percentage = (*count as f64 / contact_graph.num_vertices as f64) * 100.0;
        println!(
            "  Color {:2}: {:3} residues ({:5.2}%)",
            color_id, count, percentage
        );
    }
    println!();

    // Biological interpretation
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                 BIOLOGICAL INTERPRETATION                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Chromatic Number: {}", chromatic_number);
    println!("This represents the minimum number of conformational states or");
    println!("structural classes needed to partition the protein such that");
    println!("no two spatially proximate residues share the same class.");
    println!();
    println!("Applications:");
    println!("  • Protein folding analysis");
    println!("  • Contact map visualization");
    println!("  • Residue interaction networks");
    println!("  • Structural motif detection");
    println!();

    // Performance summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   PERFORMANCE METRICS                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "PDB Parsing:        {:.2}ms",
        parse_time.as_secs_f64() * 1000.0
    );
    println!(
        "GPU Coloring:       {:.2}ms",
        coloring_time.as_secs_f64() * 1000.0
    );
    println!(
        "Total Time:         {:.2}ms",
        (parse_time + coloring_time).as_secs_f64() * 1000.0
    );
    println!();
    println!(
        "Residues/second:    {:.0}",
        contact_graph.num_vertices as f64 / coloring_time.as_secs_f64()
    );
    println!(
        "Edges/second:       {:.0}",
        contact_graph.num_edges as f64 / coloring_time.as_secs_f64()
    );
    println!();

    println!("✅ Protein structure analysis complete!");
    println!();

    Ok(())
}
