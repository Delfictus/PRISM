// Standalone protein parser test
mod protein_parser;

use protein_parser::*;

fn main() -> anyhow::Result<()> {
    println!("=== PRISM Protein Parser - Standalone Test ===\n");

    // Check for PDB file
    let pdb_path = std::env::args().nth(1).unwrap_or_else(|| "../data/nipah/2VSM.pdb".to_string());

    if !std::path::Path::new(&pdb_path).exists() {
        eprintln!("❌ PDB file not found: {}", pdb_path);
        eprintln!("\nUsage: {} <pdb_file>", std::env::args().next().unwrap());
        eprintln!("\nExample:");
        eprintln!("  wget https://files.rcsb.org/download/2VSM.pdb");
        eprintln!("  {} ../data/nipah/2VSM.pdb", std::env::args().next().unwrap());
        return Ok(());
    }

    println!("📂 Parsing: {}", pdb_path);
    let start = std::time::Instant::now();

    let contact_graph = protein_parser::parse_pdb_file(std::path::Path::new(&pdb_path))?;

    let elapsed = start.elapsed();

    println!("\n✅ Successfully parsed protein structure in {:.2}ms!\n", elapsed.as_secs_f64() * 1000.0);

    // Display statistics
    println!("═══════════════════════════════════════════");
    println!("PROTEIN STRUCTURE STATISTICS");
    println!("═══════════════════════════════════════════");
    println!("Total residues:        {}", contact_graph.residues.len());
    println!("Residue contacts:      {}", contact_graph.contacts.len());
    println!("Distance threshold:    {:.2} Å", contact_graph.distance_threshold);

    // Calculate graph density
    let n = contact_graph.residues.len();
    if n > 1 {
        let max_edges = n * (n - 1) / 2;
        let density = (contact_graph.contacts.len() as f64 / max_edges as f64) * 100.0;
        println!("Graph density:         {:.2}%", density);
    }

    println!("\n═══════════════════════════════════════════");
    println!("FIRST 10 RESIDUES");
    println!("═══════════════════════════════════════════");
    for (i, residue) in contact_graph.residues.iter().take(10).enumerate() {
        print!("{:3}. Chain {} Residue {:4} ({:3})",
            i + 1, residue.chain, residue.residue_number, residue.residue_name);
        if let Some((x, y, z)) = residue.ca_coords {
            println!(" - Cα: ({:7.3}, {:7.3}, {:7.3})", x, y, z);
        } else {
            println!(" - No Cα coordinates");
        }
    }

    println!("\n═══════════════════════════════════════════");
    println!("CONTACT GRAPH ANALYSIS");
    println!("═══════════════════════════════════════════");

    // Analyze contact distribution
    let mut contact_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for &(r1, r2) in &contact_graph.contacts {
        *contact_counts.entry(r1).or_insert(0) += 1;
        *contact_counts.entry(r2).or_insert(0) += 1;
    }

    if !contact_counts.is_empty() {
        let max_contacts = contact_counts.values().max().copied().unwrap_or(0);
        let avg_contacts = contact_counts.values().sum::<usize>() as f64 / contact_counts.len() as f64;

        println!("Residues with contacts: {}", contact_counts.len());
        println!("Max contacts/residue:   {}", max_contacts);
        println!("Avg contacts/residue:   {:.2}", avg_contacts);

        // Find most connected residue
        if let Some((&residue_idx, &count)) = contact_counts.iter().max_by_key(|(_, &c)| c) {
            if residue_idx < contact_graph.residues.len() {
                let residue = &contact_graph.residues[residue_idx];
                println!("\nMost connected residue:");
                println!("  Chain {} Residue {} ({}) with {} contacts",
                    residue.chain, residue.residue_number, residue.residue_name, count);
            }
        }
    }

    println!("\n═══════════════════════════════════════════");
    println!("FIRST 15 CONTACTS");
    println!("═══════════════════════════════════════════");
    for (i, &(r1, r2)) in contact_graph.contacts.iter().take(15).enumerate() {
        if r1 < contact_graph.residues.len() && r2 < contact_graph.residues.len() {
            let res1 = &contact_graph.residues[r1];
            let res2 = &contact_graph.residues[r2];
            println!("{:3}. {} {}{:4} ↔ {} {}{:4}",
                i + 1,
                res1.chain, res1.residue_name, res1.residue_number,
                res2.chain, res2.residue_name, res2.residue_number);
        }
    }

    println!("\n═══════════════════════════════════════════");
    println!("✅ Protein contact graph ready for PRCT chromatic coloring!");
    println!("═══════════════════════════════════════════\n");

    Ok(())
}
