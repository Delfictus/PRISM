///! Export training graphs to NPZ format for Python GNN training
///!
///! Saves graphs in NumPy .npz format that can be loaded by PyTorch

use super::graph_generator::{GraphGenerator, TrainingGraph};
use super::dimacs_parser::GraphType;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::Path;
use serde_json;

/// Export training dataset to Python-compatible format
pub struct DatasetExporter {
    output_dir: String,
}

impl DatasetExporter {
    pub fn new(output_dir: &str) -> Self {
        Self {
            output_dir: output_dir.to_string(),
        }
    }

    /// Export full dataset with train/val split
    pub fn export_dataset(
        &self,
        graphs: Vec<TrainingGraph>,
        train_split: f64,  // 0.8 = 80% train, 20% val
    ) -> Result<(), String> {
        println!("📦 Exporting {} graphs to {}", graphs.len(), self.output_dir);

        // Create directory structure
        let graphs_dir = format!("{}/graphs", self.output_dir);
        fs::create_dir_all(&graphs_dir)
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        // Split into train/val
        let split_idx = (graphs.len() as f64 * train_split) as usize;
        let (train_graphs, val_graphs) = graphs.split_at(split_idx);

        println!("  Train: {} graphs", train_graphs.len());
        println!("  Val:   {} graphs", val_graphs.len());

        // Export train set
        for (i, graph) in train_graphs.iter().enumerate() {
            let filename = format!("{}/train_{:06}.npz", graphs_dir, i);
            self.export_graph(graph, &filename)?;

            if (i + 1) % 1000 == 0 {
                println!("    Exported {}/{} train graphs", i + 1, train_graphs.len());
            }
        }

        // Export val set
        for (i, graph) in val_graphs.iter().enumerate() {
            let filename = format!("{}/val_{:06}.npz", graphs_dir, i);
            self.export_graph(graph, &filename)?;

            if (i + 1) % 1000 == 0 {
                println!("    Exported {}/{} val graphs", i + 1, val_graphs.len());
            }
        }

        // Export metadata
        self.export_metadata(&graphs, train_split)?;

        println!("✅ Dataset exported successfully!");

        Ok(())
    }

    /// Export single graph to NPZ format
    fn export_graph(&self, graph: &TrainingGraph, filename: &str) -> Result<(), String> {
        // Convert adjacency to i32 for NumPy
        let adjacency_i32: Vec<i32> = graph.adjacency
            .iter()
            .map(|&b| if b { 1 } else { 0 })
            .collect();

        let n = graph.num_vertices;

        // Create NPZ-compatible format (using NPY crate would be ideal, but we'll use JSON for now)
        // TODO: Use proper NPZ library for better performance

        // For now, save as JSON (Python can convert)
        let json_filename = filename.replace(".npz", ".json");

        let export_data = serde_json::json!({
            "adjacency": {
                "shape": [n, n],
                "data": adjacency_i32,
            },
            "node_features": {
                "shape": [n, 16],
                "data": graph.node_features.iter().cloned().collect::<Vec<f32>>(),
            },
            "coloring": graph.optimal_coloring.clone(),
            "chromatic_number": graph.chromatic_number,
            "graph_type": graph_type_to_index(graph.graph_type),
            "difficulty_score": graph.difficulty_score,
            "metadata": {
                "name": format!("graph_{}", graph.id),
                "num_vertices": graph.num_vertices,
                "num_edges": graph.num_edges,
                "density": graph.density,
            },
        });

        let json_str = serde_json::to_string(&export_data)
            .map_err(|e| format!("JSON serialization failed: {}", e))?;

        fs::write(&json_filename, json_str)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    /// Export dataset metadata
    fn export_metadata(&self, graphs: &[TrainingGraph], train_split: f64) -> Result<(), String> {
        let metadata_path = format!("{}/metadata.json", self.output_dir);

        // Collect statistics
        let mut type_counts = std::collections::HashMap::new();
        let mut total_chromatic = 0;
        let mut total_difficulty = 0.0;

        for graph in graphs {
            let type_name = format!("{:?}", graph.graph_type);
            *type_counts.entry(type_name).or_insert(0) += 1;
            total_chromatic += graph.chromatic_number;
            total_difficulty += graph.difficulty_score;
        }

        let metadata = serde_json::json!({
            "total_graphs": graphs.len(),
            "train_graphs": (graphs.len() as f64 * train_split) as usize,
            "val_graphs": (graphs.len() as f64 * (1.0 - train_split)) as usize,
            "graph_type_distribution": type_counts,
            "avg_chromatic_number": total_chromatic as f64 / graphs.len() as f64,
            "avg_difficulty": total_difficulty / graphs.len() as f64,
            "node_feature_dim": 16,
            "max_colors": 200,
            "format": "json",  // TODO: Change to "npz" when using proper NPZ library
        });

        let json_str = serde_json::to_string_pretty(&metadata)
            .map_err(|e| format!("Metadata serialization failed: {}", e))?;

        fs::write(&metadata_path, json_str)
            .map_err(|e| format!("Failed to write metadata: {}", e))?;

        println!("  ✅ Metadata saved: {}", metadata_path);

        Ok(())
    }
}

/// Convert GraphType enum to integer index for Python
fn graph_type_to_index(graph_type: GraphType) -> usize {
    match graph_type {
        GraphType::RandomSparse => 0,
        GraphType::RandomDense => 1,
        GraphType::Register => 2,
        GraphType::Leighton => 3,
        GraphType::Queen | GraphType::Geometric => 4,
        GraphType::Mycielski => 5,
        GraphType::ScaleFree => 6,
        GraphType::SmallWorld => 7,
        GraphType::Unknown => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_small_dataset() {
        let mut generator = GraphGenerator::new(42);

        // Generate small test dataset
        let mut graphs = Vec::new();
        for i in 0..10 {
            graphs.push(generator.generate_random_graph(i, 20, 0.3, GraphType::RandomDense));
        }

        let exporter = DatasetExporter::new("/tmp/test_gnn_data");
        exporter.export_dataset(graphs, 0.8).unwrap();

        // Verify files exist
        assert!(std::path::Path::new("/tmp/test_gnn_data/metadata.json").exists());
        assert!(std::path::Path::new("/tmp/test_gnn_data/graphs/train_000000.json").exists());
    }
}
