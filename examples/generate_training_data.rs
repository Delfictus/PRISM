///! Generate 15,000 training graphs and export for GNN training
///!
///! This will take ~30-60 minutes to generate all graphs with ground truth colorings.
use prism_ai::data::{DatasetExporter, GraphGenerator};

fn main() {
    println!("🎯 PRISM-AI Graph Coloring Training Data Generation");
    println!("{}", "=".repeat(80));

    // Configuration
    let seed = 42;
    let num_graphs = 15_000;
    let output_dir = "../training_data";
    let train_split = 0.8; // 80% train, 20% val

    println!("\nConfiguration:");
    println!("  Total graphs:     {}", num_graphs);
    println!("  Train split:      {:.0}%", train_split * 100.0);
    println!("  Validation split: {:.0}%", (1.0 - train_split) * 100.0);
    println!("  Output directory: {}", output_dir);
    println!("  Random seed:      {}", seed);

    // Initialize generator
    println!("\n[1/2] Generating {} graphs...", num_graphs);
    println!("{}", "-".repeat(80));

    let mut generator = GraphGenerator::new(seed);
    let graphs = generator.generate_dataset();

    assert_eq!(
        graphs.len(),
        num_graphs,
        "Expected {} graphs, got {}",
        num_graphs,
        graphs.len()
    );

    println!("\n✅ Graph generation complete!");
    println!("  Total: {} graphs", graphs.len());

    // Analyze dataset
    println!("\n📊 Dataset Statistics:");
    println!("{}", "-".repeat(80));

    let total_vertices: usize = graphs.iter().map(|g| g.num_vertices).sum();
    let total_edges: usize = graphs.iter().map(|g| g.num_edges).sum();
    let avg_vertices = total_vertices as f64 / graphs.len() as f64;
    let avg_edges = total_edges as f64 / graphs.len() as f64;

    let min_chromatic = graphs.iter().map(|g| g.chromatic_number).min().unwrap();
    let max_chromatic = graphs.iter().map(|g| g.chromatic_number).max().unwrap();
    let avg_chromatic =
        graphs.iter().map(|g| g.chromatic_number).sum::<usize>() as f64 / graphs.len() as f64;

    let avg_difficulty =
        graphs.iter().map(|g| g.difficulty_score).sum::<f64>() / graphs.len() as f64;

    println!("  Average vertices:        {:.1}", avg_vertices);
    println!("  Average edges:           {:.1}", avg_edges);
    println!(
        "  Average degree:          {:.2}",
        graphs.iter().map(|g| g.avg_degree).sum::<f64>() / graphs.len() as f64
    );
    println!("\n  Chromatic numbers:");
    println!("    Min:     {}", min_chromatic);
    println!("    Max:     {}", max_chromatic);
    println!("    Average: {:.2}", avg_chromatic);
    println!("\n  Average difficulty:      {:.1}/100", avg_difficulty);

    // Export to Python format
    println!("\n[2/2] Exporting to Python-compatible format...");
    println!("{}", "-".repeat(80));

    let exporter = DatasetExporter::new(output_dir);

    match exporter.export_dataset(graphs, train_split) {
        Ok(_) => {
            println!("\n{}", "=".repeat(80));
            println!("✅ SUCCESS: Training data ready for GNN training!");
            println!("{}", "=".repeat(80));
            println!("\nNext steps:");
            println!("  1. cd ../python/gnn");
            println!("  2. python train.py");
            println!();
        }
        Err(e) => {
            eprintln!("\n❌ Export failed: {}", e);
            std::process::exit(1);
        }
    }
}
