//! Data handling modules for PRISM-AI
//!
//! This module provides utilities for loading, parsing, and generating
//! graph datasets for training and benchmarking.

pub mod dimacs_parser;
pub mod graph_generator;
pub mod export_training_data;

pub use dimacs_parser::{
    DimacsGraph, GraphCharacteristics, StrategyMix,
    DensityClass, GraphType,
};

pub use graph_generator::{
    GraphGenerator, TrainingGraph,
};

pub use export_training_data::{
    DatasetExporter,
};
