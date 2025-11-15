//! DIMACS Graph File Parser with Adaptive Graph Characterization
//!
//! Parses DIMACS .col format and automatically analyzes graph structure
//! to recommend optimal solving strategies for PRISM-AI Phase 6.
//!
//! Graph Types Supported:
//! - Random sparse/dense (DSJC, DSJR)
//! - Register allocation (register graphs)
//! - Adversarial (Leighton graphs - le450)
//! - Geometric (queen, myciel)
//! - Scale-free and small-world networks

use ndarray::Array2;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Density classification for strategic tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DensityClass {
    VerySparse, // < 0.05
    Sparse,     // 0.05 - 0.2
    Medium,     // 0.2 - 0.6
    Dense,      // 0.6 - 0.9
    VeryDense,  // > 0.9
}

/// Graph type classification based on structural patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphType {
    RandomSparse,
    RandomDense,
    Register,   // Register allocation graphs
    Leighton,   // Adversarial graphs designed to fool heuristics
    Queen,      // Queen graph (geometric)
    Mycielski,  // Mycielski construction
    Geometric,  // General geometric graphs
    ScaleFree,  // Power-law degree distribution
    SmallWorld, // High clustering + low diameter
    Unknown,
}

/// Recommended strategy mix for solving this graph
#[derive(Debug, Clone)]
pub struct StrategyMix {
    /// Use Topological Data Analysis
    pub use_tda: bool,
    /// Weight for TDA contribution (α parameter)
    pub tda_weight: f64,
    /// Use GNN predictions
    pub use_gnn: bool,
    /// Confidence in GNN predictions (β parameter)
    pub gnn_confidence: f64,
    /// Exploration vs exploitation balance (0=exploit, 1=explore)
    pub exploration_vs_exploitation: f64,
    /// Temperature scaling for probabilistic search
    pub temperature_scaling: f64,
    /// Factor for parallel attempts (multiplier on base attempts)
    pub parallel_attempts_factor: f64,
}

/// Comprehensive graph characteristics for adaptive strategy selection
#[derive(Debug, Clone)]
pub struct GraphCharacteristics {
    // Density metrics
    pub edge_density: f64,
    pub density_class: DensityClass,

    // Degree distribution
    pub avg_degree: f64,
    pub max_degree: usize,
    pub degree_variance: f64,

    // Structural properties
    pub clustering_coefficient: f64,
    pub diameter_estimate: usize,
    pub transitivity: f64,

    // Chromatic bounds
    pub clique_lower_bound: usize,
    pub greedy_upper_bound: usize,
    pub degeneracy: usize,

    // Classification
    pub graph_type: GraphType,
    pub difficulty_score: f64,

    // Recommended strategy
    pub recommended_strategy: StrategyMix,
}

/// DIMACS graph representation with automatic characterization
#[derive(Debug, Clone)]
pub struct DimacsGraph {
    pub name: String,
    pub num_vertices: usize,
    pub num_edges: usize,
    pub adjacency: Array2<bool>,
    pub known_chromatic: Option<usize>,
    pub characteristics: GraphCharacteristics,
}

impl DimacsGraph {
    /// Parse DIMACS .col file and automatically characterize the graph
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
        let reader = BufReader::new(file);

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let mut num_vertices = 0;
        let mut edges = Vec::new();
        let mut known_chromatic = None;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('c') {
                // Comment or empty line
                // Check for known chromatic number in comments
                if line.contains("chromatic") || line.contains("chi") {
                    if let Some(num) = extract_number_from_comment(&line) {
                        known_chromatic = Some(num);
                    }
                }
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            match parts[0] {
                "p" => {
                    // Problem line: p edge <vertices> <edges>
                    if parts.len() < 4 || parts[1] != "edge" {
                        return Err(format!("Invalid problem line: {}", line));
                    }
                    num_vertices = parts[2]
                        .parse()
                        .map_err(|_| format!("Invalid vertex count: {}", parts[2]))?;
                    // num_edges is in parts[3] but we count actual edges below
                }
                "e" => {
                    // Edge line: e <vertex1> <vertex2>
                    if parts.len() < 3 {
                        return Err(format!("Invalid edge line: {}", line));
                    }
                    let v1: usize = parts[1]
                        .parse()
                        .map_err(|_| format!("Invalid vertex: {}", parts[1]))?;
                    let v2: usize = parts[2]
                        .parse()
                        .map_err(|_| format!("Invalid vertex: {}", parts[2]))?;

                    // DIMACS uses 1-indexed vertices
                    if v1 == 0 || v2 == 0 || v1 > num_vertices || v2 > num_vertices {
                        return Err(format!("Vertex out of range: {} or {}", v1, v2));
                    }

                    edges.push((v1 - 1, v2 - 1)); // Convert to 0-indexed
                }
                _ => {
                    // Unknown line type - skip
                }
            }
        }

        if num_vertices == 0 {
            return Err("No problem line found".to_string());
        }

        // Build adjacency matrix
        let mut adjacency = Array2::from_elem((num_vertices, num_vertices), false);
        let mut actual_edges = 0;

        for (v1, v2) in edges {
            if v1 != v2 {
                // No self-loops
                adjacency[[v1, v2]] = true;
                adjacency[[v2, v1]] = true;
                actual_edges += 1;
            }
        }

        // Characterize the graph
        let characteristics = characterize_graph(&adjacency, &name);

        Ok(DimacsGraph {
            name,
            num_vertices,
            num_edges: actual_edges,
            adjacency,
            known_chromatic,
            characteristics,
        })
    }

    /// Get degree of a vertex
    pub fn degree(&self, vertex: usize) -> usize {
        self.adjacency.row(vertex).iter().filter(|&&x| x).count()
    }

    /// Get all neighbors of a vertex
    pub fn neighbors(&self, vertex: usize) -> Vec<usize> {
        self.adjacency
            .row(vertex)
            .iter()
            .enumerate()
            .filter_map(|(i, &connected)| if connected { Some(i) } else { None })
            .collect()
    }
}

/// Automatically characterize graph and recommend strategy
fn characterize_graph(adjacency: &Array2<bool>, name: &str) -> GraphCharacteristics {
    let n = adjacency.nrows();

    // Compute density metrics
    let num_edges = count_edges(adjacency);
    let max_edges = n * (n - 1) / 2;
    let edge_density = if max_edges > 0 {
        num_edges as f64 / max_edges as f64
    } else {
        0.0
    };
    let density_class = classify_density(edge_density);

    // Compute degree distribution
    let degrees = compute_degrees(adjacency);
    let avg_degree = degrees.iter().sum::<usize>() as f64 / n as f64;
    let max_degree = *degrees.iter().max().unwrap_or(&0);
    let degree_variance = compute_variance(&degrees, avg_degree);

    // Structural properties
    let clustering_coefficient = estimate_clustering(adjacency, &degrees);
    let diameter_estimate = estimate_diameter(adjacency, n);
    let transitivity = compute_transitivity(adjacency);

    // Chromatic bounds
    let clique_lower_bound = find_large_clique(adjacency);
    let greedy_upper_bound = greedy_coloring(adjacency);
    let degeneracy = compute_degeneracy(adjacency);

    // Classify graph type
    let graph_type = classify_graph_type(
        name,
        density_class,
        degree_variance,
        avg_degree,
        clustering_coefficient,
        diameter_estimate,
    );

    // Compute difficulty score (higher = harder)
    let difficulty_score = compute_difficulty(
        n,
        edge_density,
        greedy_upper_bound,
        clique_lower_bound,
        degree_variance,
        graph_type,
    );

    // Recommend strategy based on characteristics
    let recommended_strategy = recommend_strategy(
        density_class,
        graph_type,
        difficulty_score,
        avg_degree,
        clustering_coefficient,
    );

    GraphCharacteristics {
        edge_density,
        density_class,
        avg_degree,
        max_degree,
        degree_variance,
        clustering_coefficient,
        diameter_estimate,
        transitivity,
        clique_lower_bound,
        greedy_upper_bound,
        degeneracy,
        graph_type,
        difficulty_score,
        recommended_strategy,
    }
}

/// Recommend optimal strategy mix based on graph characteristics
fn recommend_strategy(
    density_class: DensityClass,
    graph_type: GraphType,
    _difficulty: f64,
    avg_degree: f64,
    clustering: f64,
) -> StrategyMix {
    use DensityClass::*;
    use GraphType::*;

    match (density_class, graph_type) {
        // LEIGHTON: Adversarial graphs designed to fool heuristics (HIGHEST PRIORITY)
        (_, Leighton) => StrategyMix {
            use_tda: true,
            tda_weight: 0.7,
            use_gnn: false, // Don't trust GNN on adversarial inputs
            gnn_confidence: 0.2,
            exploration_vs_exploitation: 0.9, // Maximum exploration
            temperature_scaling: 3.0,
            parallel_attempts_factor: 3.0,
        },

        // GEOMETRIC: Spatial structure (Queen, Geometric graphs)
        (_, Queen | Geometric) => StrategyMix {
            use_tda: true,
            tda_weight: 0.7,
            use_gnn: true,
            gnn_confidence: 0.75,
            exploration_vs_exploitation: 0.4,
            temperature_scaling: 0.8,
            parallel_attempts_factor: 1.2,
        },

        // SMALL WORLD: High clustering, low diameter
        (_, SmallWorld) if clustering > 0.3 => StrategyMix {
            use_tda: true,
            tda_weight: 0.75,
            use_gnn: true,
            gnn_confidence: 0.8,
            exploration_vs_exploitation: 0.4,
            temperature_scaling: 0.7,
            parallel_attempts_factor: 1.0,
        },

        // SCALE-FREE: Hub-based structure
        (_, ScaleFree) if avg_degree > 10.0 => StrategyMix {
            use_tda: true,
            tda_weight: 0.6,
            use_gnn: true,
            gnn_confidence: 0.7,
            exploration_vs_exploitation: 0.6,
            temperature_scaling: 1.2,
            parallel_attempts_factor: 1.5,
        },

        // REGISTER: Structured but tricky (Medium density)
        (Medium, Register) => StrategyMix {
            use_tda: true,
            tda_weight: 0.6,
            use_gnn: true,
            gnn_confidence: 0.7,
            exploration_vs_exploitation: 0.5,
            temperature_scaling: 1.0,
            parallel_attempts_factor: 1.5,
        },

        // SPARSE GRAPHS: Limited structure, high exploration needed
        (VerySparse | Sparse, _) => StrategyMix {
            use_tda: false, // Not enough edges for meaningful topology
            tda_weight: 0.2,
            use_gnn: true,
            gnn_confidence: 0.6,
            exploration_vs_exploitation: 0.8, // High exploration
            temperature_scaling: 2.0,
            parallel_attempts_factor: 2.0,
        },

        // DENSE GRAPHS: Rich structure, TDA excels here
        (Dense | VeryDense, _) => StrategyMix {
            use_tda: true,
            tda_weight: 0.8,
            use_gnn: true,
            gnn_confidence: 0.8,
            exploration_vs_exploitation: 0.3, // More exploitation
            temperature_scaling: 0.5,
            parallel_attempts_factor: 1.0,
        },

        // MEDIUM DENSITY: Balanced approach (catch remaining medium density graphs)
        (Medium, _) => StrategyMix {
            use_tda: true,
            tda_weight: 0.5,
            use_gnn: true,
            gnn_confidence: 0.7,
            exploration_vs_exploitation: 0.5,
            temperature_scaling: 1.0,
            parallel_attempts_factor: 1.5,
        },
    }
}

/// Classify graph type based on name and structural properties
fn classify_graph_type(
    name: &str,
    density: DensityClass,
    degree_variance: f64,
    avg_degree: f64,
    clustering: f64,
    diameter: usize,
) -> GraphType {
    let name_lower = name.to_lowercase();

    // Name-based classification (highest priority)
    if name_lower.starts_with("le") || name_lower.contains("leighton") {
        return GraphType::Leighton;
    }
    if name_lower.starts_with("queen") {
        return GraphType::Queen;
    }
    if name_lower.starts_with("myciel") {
        return GraphType::Mycielski;
    }
    if name_lower.starts_with("dsjr") || name_lower.contains("register") {
        return GraphType::Register;
    }
    if name_lower.starts_with("dsjc") {
        return match density {
            DensityClass::VerySparse | DensityClass::Sparse => GraphType::RandomSparse,
            _ => GraphType::RandomDense,
        };
    }

    // Structure-based classification
    // Scale-free: high degree variance (power-law distribution)
    let cv = if avg_degree > 0.0 {
        (degree_variance.sqrt()) / avg_degree
    } else {
        0.0
    };

    if cv > 1.5 {
        // High coefficient of variation suggests scale-free
        return GraphType::ScaleFree;
    }

    // Small-world: high clustering + low diameter
    if clustering > 0.3 && diameter < 10 {
        return GraphType::SmallWorld;
    }

    // Geometric graphs tend to have moderate clustering
    if clustering > 0.2 && clustering < 0.6 {
        return GraphType::Geometric;
    }

    // Fall back to density-based classification
    match density {
        DensityClass::VerySparse | DensityClass::Sparse => GraphType::RandomSparse,
        DensityClass::Dense | DensityClass::VeryDense => GraphType::RandomDense,
        _ => GraphType::Unknown,
    }
}

/// Compute difficulty score (0-100, higher = harder)
fn compute_difficulty(
    n: usize,
    density: f64,
    upper_bound: usize,
    lower_bound: usize,
    degree_variance: f64,
    graph_type: GraphType,
) -> f64 {
    let mut score = 0.0;

    // Size factor (larger = harder)
    score += (n as f64 / 1000.0).min(20.0);

    // Chromatic gap (larger gap = harder)
    let gap = upper_bound.saturating_sub(lower_bound) as f64;
    score += (gap / 5.0).min(30.0);

    // Density factor (medium density is hardest)
    let density_difficulty = if density < 0.1 {
        10.0 * density // Sparse is somewhat easy
    } else if density > 0.8 {
        10.0 * (1.0 - density) // Very dense is somewhat easy
    } else {
        20.0 // Medium density is hardest
    };
    score += density_difficulty;

    // Degree variance (higher = more structure to exploit)
    score -= (degree_variance / 100.0).min(10.0);

    // Graph type modifier
    score += match graph_type {
        GraphType::Leighton => 25.0, // Adversarial - very hard
        GraphType::Register => 15.0,
        GraphType::RandomDense => 10.0,
        GraphType::RandomSparse => 12.0,
        GraphType::ScaleFree => 8.0,
        GraphType::SmallWorld => 5.0,
        GraphType::Queen | GraphType::Geometric => 5.0,
        GraphType::Mycielski => 20.0,
        GraphType::Unknown => 10.0,
    };

    score.max(0.0).min(100.0)
}

/// Helper functions for graph analysis

fn count_edges(adjacency: &Array2<bool>) -> usize {
    let mut count = 0;
    for i in 0..adjacency.nrows() {
        for j in (i + 1)..adjacency.ncols() {
            if adjacency[[i, j]] {
                count += 1;
            }
        }
    }
    count
}

fn compute_degrees(adjacency: &Array2<bool>) -> Vec<usize> {
    (0..adjacency.nrows())
        .map(|i| adjacency.row(i).iter().filter(|&&x| x).count())
        .collect()
}

fn compute_variance(values: &[usize], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_sq_diff: f64 = values
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum();
    sum_sq_diff / values.len() as f64
}

fn classify_density(density: f64) -> DensityClass {
    if density < 0.05 {
        DensityClass::VerySparse
    } else if density < 0.2 {
        DensityClass::Sparse
    } else if density < 0.6 {
        DensityClass::Medium
    } else if density < 0.9 {
        DensityClass::Dense
    } else {
        DensityClass::VeryDense
    }
}

fn estimate_clustering(adjacency: &Array2<bool>, degrees: &[usize]) -> f64 {
    let n = adjacency.nrows();
    let mut total_clustering = 0.0;
    let mut count = 0;

    for i in 0..n {
        let degree = degrees[i];
        if degree < 2 {
            continue;
        }

        // Get neighbors
        let neighbors: Vec<usize> = (0..n).filter(|&j| adjacency[[i, j]]).collect();

        // Count triangles
        let mut triangles = 0;
        for j in 0..neighbors.len() {
            for k in (j + 1)..neighbors.len() {
                if adjacency[[neighbors[j], neighbors[k]]] {
                    triangles += 1;
                }
            }
        }

        let max_triangles = degree * (degree - 1) / 2;
        if max_triangles > 0 {
            total_clustering += triangles as f64 / max_triangles as f64;
            count += 1;
        }
    }

    if count > 0 {
        total_clustering / count as f64
    } else {
        0.0
    }
}

fn estimate_diameter(adjacency: &Array2<bool>, n: usize) -> usize {
    if n == 0 {
        return 0;
    }

    // Sample BFS from a few vertices to estimate diameter
    let sample_size = n.min(10);
    let mut max_distance = 0;

    for start in (0..n).step_by(n / sample_size.max(1)) {
        let distances = bfs_distances(adjacency, start);
        if let Some(&max_dist) = distances.iter().max() {
            if max_dist < usize::MAX {
                max_distance = max_distance.max(max_dist);
            }
        }
    }

    max_distance
}

fn bfs_distances(adjacency: &Array2<bool>, start: usize) -> Vec<usize> {
    let n = adjacency.nrows();
    let mut distances = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    distances[start] = 0;
    queue.push_back(start);

    while let Some(u) = queue.pop_front() {
        for v in 0..n {
            if adjacency[[u, v]] && distances[v] == usize::MAX {
                distances[v] = distances[u] + 1;
                queue.push_back(v);
            }
        }
    }

    distances
}

fn compute_transitivity(adjacency: &Array2<bool>) -> f64 {
    let n = adjacency.nrows();
    let mut triangles = 0;
    let mut triples = 0;

    for i in 0..n {
        let neighbors: Vec<usize> = (0..n).filter(|&j| adjacency[[i, j]]).collect();

        let degree = neighbors.len();
        if degree < 2 {
            continue;
        }

        triples += degree * (degree - 1) / 2;

        for j in 0..neighbors.len() {
            for k in (j + 1)..neighbors.len() {
                if adjacency[[neighbors[j], neighbors[k]]] {
                    triangles += 1;
                }
            }
        }
    }

    if triples > 0 {
        triangles as f64 / triples as f64
    } else {
        0.0
    }
}

fn find_large_clique(adjacency: &Array2<bool>) -> usize {
    let n = adjacency.nrows();
    let mut max_clique = 1;

    // Greedy clique finding - not optimal but fast
    let degrees = compute_degrees(adjacency);
    let mut vertices: Vec<usize> = (0..n).collect();
    vertices.sort_by_key(|&v| std::cmp::Reverse(degrees[v]));

    for &start in vertices.iter().take(20.min(n)) {
        let mut clique = vec![start];

        for &candidate in &vertices {
            if candidate == start {
                continue;
            }

            // Check if candidate is connected to all in clique
            if clique.iter().all(|&v| adjacency[[candidate, v]]) {
                clique.push(candidate);
            }
        }

        max_clique = max_clique.max(clique.len());
    }

    max_clique
}

fn greedy_coloring(adjacency: &Array2<bool>) -> usize {
    let n = adjacency.nrows();
    let mut colors = vec![None; n];

    // Order by degree (descending)
    let degrees = compute_degrees(adjacency);
    let mut vertices: Vec<usize> = (0..n).collect();
    vertices.sort_by_key(|&v| std::cmp::Reverse(degrees[v]));

    for &v in &vertices {
        // Find used neighbor colors
        let mut used_colors = HashSet::new();
        for u in 0..n {
            if adjacency[[v, u]] {
                if let Some(color) = colors[u] {
                    used_colors.insert(color);
                }
            }
        }

        // Assign smallest available color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }
        colors[v] = Some(color);
    }

    // Return number of colors used
    colors.iter().filter_map(|&c| c).max().unwrap_or(0) + 1
}

fn compute_degeneracy(adjacency: &Array2<bool>) -> usize {
    let n = adjacency.nrows();
    let mut remaining: HashSet<usize> = (0..n).collect();
    let mut max_degeneracy = 0;

    while !remaining.is_empty() {
        // Find vertex with minimum degree in remaining graph
        let (min_vertex, min_degree) = remaining
            .iter()
            .map(|&v| {
                let deg = remaining.iter().filter(|&&u| adjacency[[v, u]]).count();
                (v, deg)
            })
            .min_by_key(|&(_, deg)| deg)
            .unwrap();

        max_degeneracy = max_degeneracy.max(min_degree);
        remaining.remove(&min_vertex);
    }

    max_degeneracy
}

fn extract_number_from_comment(line: &str) -> Option<usize> {
    // Try to extract number from comment like "c chi(G) = 82"
    for word in line.split_whitespace() {
        if let Ok(num) = word.trim_matches(|c: char| !c.is_numeric()).parse() {
            return Some(num);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_classification() {
        assert_eq!(classify_density(0.01), DensityClass::VerySparse);
        assert_eq!(classify_density(0.1), DensityClass::Sparse);
        assert_eq!(classify_density(0.4), DensityClass::Medium);
        assert_eq!(classify_density(0.7), DensityClass::Dense);
        assert_eq!(classify_density(0.95), DensityClass::VeryDense);
    }

    #[test]
    fn test_graph_type_classification() {
        assert_eq!(
            classify_graph_type("le450_15a", DensityClass::Medium, 100.0, 50.0, 0.3, 5),
            GraphType::Leighton
        );
        assert_eq!(
            classify_graph_type("queen8_8", DensityClass::Sparse, 10.0, 12.0, 0.2, 8),
            GraphType::Queen
        );
        assert_eq!(
            classify_graph_type("DSJC125.1", DensityClass::Sparse, 20.0, 5.0, 0.1, 15),
            GraphType::RandomSparse
        );
    }

    #[test]
    fn test_small_graph() {
        // Create simple triangle graph
        let mut adj = Array2::from_elem((3, 3), false);
        adj[[0, 1]] = true;
        adj[[1, 0]] = true;
        adj[[1, 2]] = true;
        adj[[2, 1]] = true;
        adj[[0, 2]] = true;
        adj[[2, 0]] = true;

        let chars = characterize_graph(&adj, "triangle");

        assert_eq!(chars.density_class, DensityClass::VeryDense);
        assert_eq!(chars.clique_lower_bound, 3);
        assert_eq!(chars.greedy_upper_bound, 3);
        assert_eq!(chars.num_edges, 3);
        assert!((chars.clustering_coefficient - 1.0).abs() < 0.01);
    }
}
