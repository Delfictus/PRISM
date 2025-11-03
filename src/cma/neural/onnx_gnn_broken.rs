///! ONNX Runtime CUDA GNN Inference
///!
///! Real ONNX Runtime integration with CUDA execution provider.
///! Replaces placeholder predictions with actual neural network inference.

use ndarray::{Array1, Array2};
use ort::{Session, Value, GraphOptimizationLevel, ExecutionProvider};
use std::path::Path;
use anyhow::{Result, anyhow};

/// GNN predictions from ONNX model
#[derive(Debug, Clone)]
pub struct OnnxGnnPrediction {
    /// Color logits per node [N, max_colors]
    pub node_color_logits: Array2<f32>,

    /// Predicted chromatic number
    pub predicted_chromatic: usize,

    /// Graph type classification logits [num_types]
    pub graph_type_logits: Array1<f32>,

    /// Difficulty score [0, 100]
    pub difficulty_score: f32,

    /// Inference time (ms)
    pub inference_time_ms: f64,
}

/// ONNX Runtime GNN with CUDA execution
pub struct OnnxGNN {
    session: Session,
    max_colors: usize,
}

impl OnnxGNN {
    /// Create new ONNX GNN with CUDA execution provider
    ///
    /// # Arguments
    /// - `model_path`: Path to ONNX model file
    /// - `max_colors`: Maximum number of colors (must match training)
    ///
    /// # GPU-ONLY
    /// Uses CUDA execution provider, fails if unavailable
    pub fn new<P: AsRef<Path>>(model_path: P, max_colors: usize) -> Result<Self> {
        let path = model_path.as_ref();

        if !path.exists() {
            return Err(anyhow!("Model file not found: {}", path.display()));
        }

        println!("[ONNX-GNN] Loading model: {}", path.display());
        println!("[ONNX-GNN]   Max colors: {}", max_colors);

        // Configure ONNX Runtime with CUDA execution provider
        let session = Session::builder()?
            .with_execution_providers([
                ExecutionProvider::CUDA(Default::default()),
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(path)?;

        println!("[ONNX-GNN] ✅ Model loaded with CUDA execution provider");

        // Print input/output info
        println!("[ONNX-GNN]   Inputs:");
        for (i, input) in session.inputs.iter().enumerate() {
            println!("[ONNX-GNN]     [{}] {}: {:?}", i, input.name, input.input_type);
        }
        println!("[ONNX-GNN]   Outputs:");
        for (i, output) in session.outputs.iter().enumerate() {
            println!("[ONNX-GNN]     [{}] {}: {:?}", i, output.name, output.output_type);
        }

        Ok(Self {
            session,
            max_colors,
        })
    }

    /// Run inference on graph
    ///
    /// # Arguments
    /// - `node_features`: Node features [N, feature_dim]
    /// - `edge_index`: Edge list [2, num_edges]
    ///
    /// # Returns
    /// GNN predictions for graph coloring
    pub fn predict(
        &self,
        node_features: &Array2<f32>,
        edge_index: &Array2<i64>,
    ) -> Result<OnnxGnnPrediction> {
        let n = node_features.nrows();
        let start = std::time::Instant::now();

        println!("[ONNX-GNN] Running inference: {} nodes, {} edges",
                 n, edge_index.ncols());

        // Prepare inputs as ONNX values
        let node_features_value = Value::from_array(self.session.allocator(), node_features)?;
        let edge_index_value = Value::from_array(self.session.allocator(), edge_index)?;

        // Create batch tensor (all nodes in batch 0)
        let batch = Array1::<i64>::zeros(n);
        let batch_value = Value::from_array(self.session.allocator(), &batch)?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "node_features" => node_features_value,
            "edge_index" => edge_index_value,
            "batch" => batch_value,
        ]?)?;

        let inference_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Extract outputs
        // Output 0: node_color_logits [N, max_colors]
        let node_color_logits_value = &outputs[0];
        let node_color_logits: Array2<f32> = node_color_logits_value.try_extract_tensor()?.to_owned();

        // Output 1: chromatic_number_logits [1, max_chromatic_range]
        let chromatic_value = &outputs[1];
        let chromatic_logits: Array2<f32> = chromatic_value.try_extract_tensor()?.to_owned();
        let predicted_chromatic = chromatic_logits.row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i + 1)  // +1 because chromatic starts at 1
            .unwrap_or(1);

        // Output 2: graph_type_logits [1, num_types]
        let graph_type_value = &outputs[2];
        let graph_type_2d: Array2<f32> = graph_type_value.try_extract_tensor()?.to_owned();
        let graph_type_logits = graph_type_2d.row(0).to_owned();

        // Output 3: difficulty_score [1]
        let difficulty_value = &outputs[3];
        let difficulty_arr: Array1<f32> = difficulty_value.try_extract_tensor()?.to_owned();
        let difficulty_score = difficulty_arr[0];

        println!("[ONNX-GNN] ✅ Inference complete: {:.2}ms", inference_time_ms);
        println!("[ONNX-GNN]   Predicted chromatic: {}", predicted_chromatic);
        println!("[ONNX-GNN]   Difficulty: {:.1}", difficulty_score);

        Ok(OnnxGnnPrediction {
            node_color_logits,
            predicted_chromatic,
            graph_type_logits,
            difficulty_score,
            inference_time_ms,
        })
    }

    /// Check if model file exists at path
    pub fn model_exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]  // Requires trained model file
    fn test_onnx_gnn_loading() {
        let model_path = "../models/coloring_gnn.onnx";

        if !OnnxGNN::model_exists(model_path) {
            println!("⚠️  Skipping test - model not found: {}", model_path);
            return;
        }

        let gnn = OnnxGNN::new(model_path, 50).expect("Failed to load model");
        println!("✅ ONNX GNN loaded successfully");

        // Test inference on small graph
        let node_features = Array2::from_shape_fn((3, 16), |(i, j)| {
            (i + j) as f32 * 0.1
        });

        let edge_index = Array2::from_shape_vec((2, 2), vec![
            0, 1,  // Edge 0->1
            1, 2,  // Edge 1->2
        ]).unwrap();

        let prediction = gnn.predict(&node_features, &edge_index)
            .expect("Inference failed");

        assert_eq!(prediction.node_color_logits.nrows(), 3);
        assert!(prediction.predicted_chromatic > 0);
        println!("✅ Inference test passed");
    }
}
