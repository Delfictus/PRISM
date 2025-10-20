use anyhow::Result;
use prism_ai::cuda::gpu_coloring::GpuColoringEngine;
///! Test DSJC1000.5 with dynamic memory CUDA implementation
///!
///! Validates that the GPU coloring with dynamic memory works correctly
///! for large graphs (1000 vertices).
use prism_ai::data::DimacsGraph;
use std::path::Path;

fn validate_coloring(adjacency: &ndarray::Array2<bool>, coloring: &[usize]) -> bool {
    let n = adjacency.nrows();
    if coloring.len() != n {
        println!("❌ Coloring has wrong size: {} vs {}", coloring.len(), n);
        return false;
    }

    // Check that no two adjacent vertices have the same color
    let mut violations = 0;
    for i in 0..n {
        for j in i + 1..n {
            if adjacency[[i, j]] && coloring[i] == coloring[j] {
                violations += 1;
                if violations <= 5 {
                    println!(
                        "  ❌ Vertices {} and {} are adjacent but both have color {}",
                        i, j, coloring[i]
                    );
                }
            }
        }
    }

    if violations > 0 {
        println!("❌ Total violations: {}", violations);
        return false;
    }

    true
}

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  DSJC1000.5 Dynamic Memory Test              ║");
    println!("║  Testing 1000-vertex graph with GPU coloring ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Load DSJC1000.5
    let graph_path = Path::new("../benchmarks/dimacs/DSJC1000.5.col");

    println!("📊 Loading DSJC1000.5...");
    let graph = DimacsGraph::from_file(graph_path)
        .map_err(|e| anyhow::anyhow!("Failed to load graph: {}", e))?;

    println!("✅ Graph loaded:");
    println!("   Vertices: {}", graph.num_vertices);
    println!("   Edges: {}", graph.num_edges);
    println!(
        "   Density: {:.1}%",
        graph.characteristics.edge_density * 100.0
    );
    // Best known is 82 colors for DSJC1000.5
    println!("   Best known: 82 colors (world record)");
    println!();

    // Initialize GPU coloring engine
    println!("🚀 Initializing GPU coloring engine...");
    let gpu_engine = match GpuColoringEngine::new() {
        Ok(engine) => {
            println!("✅ GPU engine ready");
            engine
        }
        Err(e) => {
            println!("❌ Failed to initialize GPU: {}", e);
            println!("   Requires CUDA-capable device");
            return Ok(());
        }
    };

    // Run GPU coloring with different configurations
    println!("\n🎨 Running GPU coloring tests...\n");

    // Test 1: Quick test with few attempts
    println!("Test 1: Quick validation (100 attempts)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let result1 = gpu_engine.color_graph(
        &graph.adjacency,
        100,  // num_attempts
        1.0,  // temperature
        1000, // max_colors
    )?;

    println!("   Chromatic: {} colors", result1.chromatic_number);
    println!("   Runtime: {:.2}ms", result1.runtime_ms);

    let valid1 = validate_coloring(&graph.adjacency, &result1.coloring);
    println!("   Valid: {}", if valid1 { "✅ YES" } else { "❌ NO" });

    // Test 2: More attempts for better quality
    println!("\nTest 2: Quality run (1000 attempts)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    let result2 = gpu_engine.color_graph(
        &graph.adjacency,
        1000, // num_attempts
        1.5,  // temperature (higher for more exploration)
        1000, // max_colors
    )?;

    println!("   Chromatic: {} colors", result2.chromatic_number);
    println!("   Runtime: {:.2}ms", result2.runtime_ms);

    let valid2 = validate_coloring(&graph.adjacency, &result2.coloring);
    println!("   Valid: {}", if valid2 { "✅ YES" } else { "❌ NO" });

    // Show statistics
    println!("\n📈 Statistics from 1000 attempts:");
    println!(
        "   Min chromatic: {}",
        result2.all_attempts.iter().min().unwrap()
    );
    println!(
        "   Max chromatic: {}",
        result2.all_attempts.iter().max().unwrap()
    );
    println!(
        "   Avg chromatic: {:.1}",
        result2.all_attempts.iter().sum::<usize>() as f64 / result2.all_attempts.len() as f64
    );

    // Final summary
    println!("\n╔══════════════════════════════════════╗");
    println!("║           TEST SUMMARY                ║");
    println!("╠══════════════════════════════════════╣");

    if valid1 && valid2 {
        println!("║ ✅ ALL TESTS PASSED!                  ║");
        println!("║                                       ║");
        println!("║ Dynamic memory implementation works   ║");
        println!("║ correctly for 1000-vertex graphs!    ║");
        println!("║                                       ║");
        println!(
            "║ Best result: {} colors               ║",
            result1.chromatic_number.min(result2.chromatic_number)
        );

        let best_known = 82; // World record for DSJC1000.5
        let best_result = result1.chromatic_number.min(result2.chromatic_number);

        if best_result <= best_known {
            println!("║ 🏆 MATCHED/BEAT WORLD RECORD!        ║");
        } else {
            let gap = best_result - best_known;
            println!("║ Gap to record: {} colors             ║", gap);
        }
    } else {
        println!("║ ❌ VALIDATION FAILED                  ║");
        println!("║                                       ║");
        println!("║ Coloring contains conflicts.         ║");
        println!("║ Further debugging required.          ║");
    }

    println!("╚══════════════════════════════════════╝");

    Ok(())
}
