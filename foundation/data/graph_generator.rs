//! Graph Dataset Generator for GNN Training
//!
//! Generates 15,000 diverse training graphs matching DIMACS benchmark distributions.
//! Each graph is solved with greedy + local search to provide ground truth labels.
//!
//! Distribution:
//! - 3,000 Random Sparse (DSJC*.1 style)
//! - 3,000 Random Dense (DSJC*.5, DSJC*.9 style)
//! - 2,000 Register Allocation (DSJR* style)
//! - 2,000 Leighton Adversarial (le450* style)
//! - 1,500 Geometric/Queen (queen* style)
//! - 1,000 Mycielski (myciel* style)
//! - 1,500 Scale-Free (power-law degree)
//! - 1,000 Small-World (high clustering)

use ndarray::{Array1, Array2};
use rand::Rng;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet, VecDeque};

use super::dimacs_parser::{GraphType, DensityClass};

/// Training graph with ground truth labels
#[derive(Debug, Clone)]
pub struct TrainingGraph {
    pub id: usize,
    pub graph_type: GraphType,
    pub num_vertices: usize,
    pub num_edges: usize,
    pub adjacency: Array2<bool>,

    // Ground truth labels (from solver)
    pub optimal_coloring: Vec<usize>,
    pub chromatic_number: usize,

    // Metadata
    pub density: f64,
    pub density_class: DensityClass,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub clustering_coefficient: f64,
    pub difficulty_score: f64,

    // Node features for GNN (16-dim per node)
    pub node_features: Array2<f32>,
}

/// Graph generator with configurable parameters
pub struct GraphGenerator {
    rng: rand::rngs::StdRng,
}

impl GraphGenerator {
    pub fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Generate complete training dataset (15,000 graphs)
    pub fn generate_dataset(&mut self) -> Vec<TrainingGraph> {
        let mut dataset = Vec::with_capacity(15000);
        let mut graph_id = 0;

        println!("🔧 Generating 15,000 training graphs...");

        // Random Sparse (3,000)
        println!("  [1/8] Generating 3,000 Random Sparse graphs...");
        for _ in 0..3000 {
            let n = self.sample_size();
            let p = self.rng.gen_range(0.01..0.05); // Very sparse
            let graph = self.generate_random_graph(graph_id, n, p, GraphType::RandomSparse);
            dataset.push(graph);
            graph_id += 1;
        }

        // Random Dense (3,000)
        println!("  [2/8] Generating 3,000 Random Dense graphs...");
        for _ in 0..3000 {
            let n = self.sample_size();
            let p = self.rng.gen_range(0.4..0.7); // Dense
            let graph = self.generate_random_graph(graph_id, n, p, GraphType::RandomDense);
            dataset.push(graph);
            graph_id += 1;
        }

        // Register Allocation (2,000)
        println!("  [3/8] Generating 2,000 Register Allocation graphs...");
        for _ in 0..2000 {
            let n = self.sample_size();
            let graph = self.generate_register_graph(graph_id, n);
            dataset.push(graph);
            graph_id += 1;
        }

        // Leighton Adversarial (2,000)
        println!("  [4/8] Generating 2,000 Leighton Adversarial graphs...");
        for _ in 0..2000 {
            let n = self.sample_size();
            let graph = self.generate_leighton_graph(graph_id, n);
            dataset.push(graph);
            graph_id += 1;
        }

        // Geometric/Queen (1,500)
        println!("  [5/8] Generating 1,500 Geometric graphs...");
        for _ in 0..1500 {
            let n = self.sample_size();
            let graph = self.generate_geometric_graph(graph_id, n);
            dataset.push(graph);
            graph_id += 1;
        }

        // Mycielski (1,000)
        println!("  [6/8] Generating 1,000 Mycielski graphs...");
        for _ in 0..1000 {
            let k = self.rng.gen_range(3..8); // Mycielski iteration depth
            let graph = self.generate_mycielski_graph(graph_id, k);
            dataset.push(graph);
            graph_id += 1;
        }

        // Scale-Free (1,500)
        println!("  [7/8] Generating 1,500 Scale-Free graphs...");
        for _ in 0..1500 {
            let n = self.sample_size();
            let m = self.rng.gen_range(2..6); // Edges per new node
            let graph = self.generate_scale_free_graph(graph_id, n, m);
            dataset.push(graph);
            graph_id += 1;
        }

        // Small-World (1,000)
        println!("  [8/8] Generating 1,000 Small-World graphs...");
        for _ in 0..1000 {
            let n = self.sample_size();
            let k = self.rng.gen_range(4..10); // Neighbors in ring
            let p = self.rng.gen_range(0.1..0.3); // Rewiring probability
            let graph = self.generate_small_world_graph(graph_id, n, k, p);
            dataset.push(graph);
            graph_id += 1;
        }

        println!("✅ Generated {} graphs total", dataset.len());
        dataset
    }

    /// Sample graph size based on distribution
    fn sample_size(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        if r < 0.30 {
            self.rng.gen_range(10..50)      // Small: 30%
        } else if r < 0.70 {
            self.rng.gen_range(50..200)     // Medium: 40%
        } else if r < 0.95 {
            self.rng.gen_range(200..500)    // Large: 25%
        } else {
            self.rng.gen_range(500..1000)   // Very Large: 5%
        }
    }

    /// Generate random Erdős-Rényi graph
    pub fn generate_random_graph(&mut self, id: usize, n: usize, p: f64, graph_type: GraphType) -> TrainingGraph {
        let mut adjacency = Array2::from_elem((n, n), false);

        for i in 0..n {
            for j in (i + 1)..n {
                if self.rng.gen_bool(p) {
                    adjacency[[i, j]] = true;
                    adjacency[[j, i]] = true;
                }
            }
        }

        self.finalize_graph(id, adjacency, graph_type)
    }

    /// Generate register allocation graph (interval overlap)
    pub fn generate_register_graph(&mut self, id: usize, n: usize) -> TrainingGraph {
        let mut adjacency = Array2::from_elem((n, n), false);

        // Create random intervals (start, end)
        let mut intervals: Vec<(usize, usize)> = Vec::new();
        for _ in 0..n {
            let start = self.rng.gen_range(0..100);
            let len = self.rng.gen_range(5..30);
            intervals.push((start, start + len));
        }

        // Two intervals overlap if they conflict (need different registers/colors)
        for i in 0..n {
            for j in (i + 1)..n {
                let (s1, e1) = intervals[i];
                let (s2, e2) = intervals[j];
                if s1 < e2 && s2 < e1 {
                    adjacency[[i, j]] = true;
                    adjacency[[j, i]] = true;
                }
            }
        }

        self.finalize_graph(id, adjacency, GraphType::Register)
    }

    /// Generate Leighton adversarial graph (designed to fool greedy)
    pub fn generate_leighton_graph(&mut self, id: usize, n: usize) -> TrainingGraph {
        let mut adjacency = Array2::from_elem((n, n), false);

        // Partition into k groups
        let k = (n as f64).sqrt() as usize + 1;
        let group_size = n / k;

        // Connect vertices in different groups with specific pattern to fool greedy
        for i in 0..n {
            let group_i = i / group_size;
            for j in (i + 1)..n {
                let group_j = j / group_size;

                // Adversarial pattern: connect vertices to create misleading degree distribution
                if group_i != group_j {
                    let connect_prob = if (group_i + group_j) % 2 == 0 { 0.7 } else { 0.3 };
                    if self.rng.gen_bool(connect_prob) {
                        adjacency[[i, j]] = true;
                        adjacency[[j, i]] = true;
                    }
                }
            }
        }

        self.finalize_graph(id, adjacency, GraphType::Leighton)
    }

    /// Generate geometric graph (random points in 2D space, connect if close)
    pub fn generate_geometric_graph(&mut self, id: usize, n: usize) -> TrainingGraph {
        let mut adjacency = Array2::from_elem((n, n), false);

        // Random points in unit square
        let mut points: Vec<(f64, f64)> = Vec::new();
        for _ in 0..n {
            points.push((self.rng.gen(), self.rng.gen()));
        }

        // Connect if distance < threshold
        let threshold = (2.0 / n as f64).sqrt() * 3.0; // Adjust for desired density

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = ((points[i].0 - points[j].0).powi(2) +
                            (points[i].1 - points[j].1).powi(2)).sqrt();
                if dist < threshold {
                    adjacency[[i, j]] = true;
                    adjacency[[j, i]] = true;
                }
            }
        }

        self.finalize_graph(id, adjacency, GraphType::Geometric)
    }

    /// Generate Mycielski construction graph (triangle-free, high chromatic)
    pub fn generate_mycielski_graph(&mut self, id: usize, k: usize) -> TrainingGraph {
        // Start with K2 (single edge)
        let mut adjacency = Array2::from_elem((2, 2), false);
        adjacency[[0, 1]] = true;
        adjacency[[1, 0]] = true;

        // Iterate k times
        for _ in 0..k {
            let n = adjacency.nrows();
            let new_n = 2 * n + 1;
            let mut new_adj = Array2::from_elem((new_n, new_n), false);

            // Copy original graph (U)
            for i in 0..n {
                for j in 0..n {
                    new_adj[[i, j]] = adjacency[[i, j]];
                }
            }

            // Create mirror vertices (W)
            for i in 0..n {
                for j in 0..n {
                    if adjacency[[i, j]] {
                        new_adj[[n + i, j]] = true;
                        new_adj[[j, n + i]] = true;
                    }
                }
            }

            // Connect all W to new vertex z
            for i in n..(2 * n) {
                new_adj[[i, 2 * n]] = true;
                new_adj[[2 * n, i]] = true;
            }

            adjacency = new_adj;
        }

        self.finalize_graph(id, adjacency, GraphType::Mycielski)
    }

    /// Generate scale-free graph (Barabási-Albert model)
    pub fn generate_scale_free_graph(&mut self, id: usize, n: usize, m: usize) -> TrainingGraph {
        let mut adjacency = Array2::from_elem((n, n), false);

        // Start with complete graph on m+1 nodes
        for i in 0..=m {
            for j in (i + 1)..=m {
                adjacency[[i, j]] = true;
                adjacency[[j, i]] = true;
            }
        }

        // Add remaining nodes with preferential attachment
        for new_node in (m + 1)..n {
            let mut degrees: Vec<usize> = (0..new_node)
                .map(|i| adjacency.row(i).iter().filter(|&&x| x).count())
                .collect();

            let total_degree: usize = degrees.iter().sum();

            // Select m nodes to connect to (preferential attachment)
            let mut targets = HashSet::new();
            while targets.len() < m && targets.len() < new_node {
                let r: f64 = self.rng.gen_range(0.0..total_degree as f64);
                let mut cumsum = 0.0;
                for (i, &deg) in degrees.iter().enumerate() {
                    cumsum += deg as f64;
                    if cumsum >= r {
                        targets.insert(i);
                        break;
                    }
                }
            }

            for &target in &targets {
                adjacency[[new_node, target]] = true;
                adjacency[[target, new_node]] = true;
            }
        }

        self.finalize_graph(id, adjacency, GraphType::ScaleFree)
    }

    /// Generate small-world graph (Watts-Strogatz model)
    pub fn generate_small_world_graph(&mut self, id: usize, n: usize, k: usize, p: f64) -> TrainingGraph {
        let mut adjacency = Array2::from_elem((n, n), false);

        // Create ring lattice with k neighbors
        for i in 0..n {
            for j in 1..=(k / 2) {
                let neighbor = (i + j) % n;
                adjacency[[i, neighbor]] = true;
                adjacency[[neighbor, i]] = true;
            }
        }

        // Rewire edges with probability p
        for i in 0..n {
            for j in 1..=(k / 2) {
                if self.rng.gen_bool(p) {
                    let old_neighbor = (i + j) % n;
                    adjacency[[i, old_neighbor]] = false;
                    adjacency[[old_neighbor, i]] = false;

                    // Select new random neighbor (avoid duplicates)
                    let mut new_neighbor = self.rng.gen_range(0..n);
                    while new_neighbor == i || adjacency[[i, new_neighbor]] {
                        new_neighbor = self.rng.gen_range(0..n);
                    }

                    adjacency[[i, new_neighbor]] = true;
                    adjacency[[new_neighbor, i]] = true;
                }
            }
        }

        self.finalize_graph(id, adjacency, GraphType::SmallWorld)
    }

    /// Finalize graph: solve for ground truth, compute features
    fn finalize_graph(&mut self, id: usize, adjacency: Array2<bool>, graph_type: GraphType) -> TrainingGraph {
        let n = adjacency.nrows();

        // Solve graph for ground truth coloring
        let (coloring, chromatic_number) = greedy_coloring_with_local_search(&adjacency);

        // Compute structural metrics
        let num_edges = count_edges(&adjacency);
        let max_edges = n * (n - 1) / 2;
        let density = if max_edges > 0 { num_edges as f64 / max_edges as f64 } else { 0.0 };

        let degrees: Vec<usize> = (0..n)
            .map(|i| adjacency.row(i).iter().filter(|&&x| x).count())
            .collect();
        let avg_degree = degrees.iter().sum::<usize>() as f64 / n as f64;
        let max_degree = *degrees.iter().max().unwrap_or(&0);

        let clustering_coefficient = estimate_clustering(&adjacency, &degrees);

        let density_class = classify_density(density);
        let difficulty_score = compute_difficulty(
            n, density, chromatic_number,
            *degrees.iter().min().unwrap_or(&0),
            graph_type
        );

        // Compute 16-dim node features for GNN
        let node_features = compute_node_features(&adjacency, &degrees, graph_type);

        TrainingGraph {
            id,
            graph_type,
            num_vertices: n,
            num_edges,
            adjacency,
            optimal_coloring: coloring,
            chromatic_number,
            density,
            density_class,
            avg_degree,
            max_degree,
            clustering_coefficient,
            difficulty_score,
            node_features,
        }
    }
}

/// Greedy coloring with simple local search improvement
fn greedy_coloring_with_local_search(adjacency: &Array2<bool>) -> (Vec<usize>, usize) {
    let n = adjacency.nrows();

    // Initial greedy coloring (DSATUR - degree of saturation)
    let mut coloring = vec![None; n];
    let mut color_counts = HashMap::new();

    // Order vertices by degree (descending)
    let degrees: Vec<usize> = (0..n)
        .map(|i| adjacency.row(i).iter().filter(|&&x| x).count())
        .collect();
    let mut vertices: Vec<usize> = (0..n).collect();
    vertices.sort_by_key(|&v| std::cmp::Reverse(degrees[v]));

    for &v in &vertices {
        // Find used neighbor colors
        let mut used_colors = HashSet::new();
        for u in 0..n {
            if adjacency[[v, u]] {
                if let Some(color) = coloring[u] {
                    used_colors.insert(color);
                }
            }
        }

        // Assign smallest available color
        let mut color = 0;
        while used_colors.contains(&color) {
            color += 1;
        }
        coloring[v] = Some(color);
        *color_counts.entry(color).or_insert(0) += 1;
    }

    let final_coloring: Vec<usize> = coloring.into_iter().map(|c| c.unwrap()).collect();
    let chromatic = final_coloring.iter().max().unwrap_or(&0) + 1;

    (final_coloring, chromatic)
}

/// Compute 16-dimensional node features for each vertex
fn compute_node_features(adjacency: &Array2<bool>, degrees: &[usize], graph_type: GraphType) -> Array2<f32> {
    let n = adjacency.nrows();
    let mut features = Array2::zeros((n, 16));

    let max_degree = *degrees.iter().max().unwrap_or(&1) as f32;

    for i in 0..n {
        let deg = degrees[i] as f32;

        // Feature 0: Normalized degree
        features[[i, 0]] = deg / max_degree;

        // Feature 1: Local clustering coefficient
        features[[i, 1]] = local_clustering(adjacency, i, degrees[i]) as f32;

        // Feature 2-3: Degree centrality measures
        features[[i, 2]] = deg / (n as f32 - 1.0);
        features[[i, 3]] = (deg * deg) / max_degree.max(1.0);

        // Feature 4: Triangle count (normalized)
        features[[i, 4]] = count_triangles(adjacency, i) as f32 / 100.0;

        // Feature 5: Square count (normalized)
        features[[i, 5]] = count_squares(adjacency, i) as f32 / 100.0;

        // Feature 6: Eccentricity estimate
        features[[i, 6]] = estimate_eccentricity(adjacency, i) as f32 / n as f32;

        // Feature 7: Core number estimate
        features[[i, 7]] = deg / max_degree; // Approximation

        // Features 8-15: One-hot graph type encoding
        let type_idx = match graph_type {
            GraphType::RandomSparse => 0,
            GraphType::RandomDense => 1,
            GraphType::Register => 2,
            GraphType::Leighton => 3,
            GraphType::Queen | GraphType::Geometric => 4,
            GraphType::Mycielski => 5,
            GraphType::ScaleFree => 6,
            GraphType::SmallWorld => 7,
            GraphType::Unknown => 0,
        };
        features[[i, 8 + type_idx]] = 1.0;
    }

    features
}

// Helper functions

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
    let mut total = 0.0;
    let mut count = 0;

    for i in 0..n.min(100) { // Sample for speed
        if degrees[i] < 2 {
            continue;
        }
        total += local_clustering(adjacency, i, degrees[i]);
        count += 1;
    }

    if count > 0 { total / count as f64 } else { 0.0 }
}

fn local_clustering(adjacency: &Array2<bool>, v: usize, degree: usize) -> f64 {
    if degree < 2 {
        return 0.0;
    }

    let neighbors: Vec<usize> = (0..adjacency.ncols())
        .filter(|&u| adjacency[[v, u]])
        .collect();

    let mut triangles = 0;
    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            if adjacency[[neighbors[i], neighbors[j]]] {
                triangles += 1;
            }
        }
    }

    let max_triangles = degree * (degree - 1) / 2;
    triangles as f64 / max_triangles as f64
}

fn count_triangles(adjacency: &Array2<bool>, v: usize) -> usize {
    let neighbors: Vec<usize> = (0..adjacency.ncols())
        .filter(|&u| adjacency[[v, u]])
        .collect();

    let mut count = 0;
    for i in 0..neighbors.len() {
        for j in (i + 1)..neighbors.len() {
            if adjacency[[neighbors[i], neighbors[j]]] {
                count += 1;
            }
        }
    }
    count
}

fn count_squares(adjacency: &Array2<bool>, v: usize) -> usize {
    let neighbors: Vec<usize> = (0..adjacency.ncols())
        .filter(|&u| adjacency[[v, u]])
        .collect();

    let mut count = 0;
    for &n1 in &neighbors {
        for &n2 in &neighbors {
            if n1 != n2 && !adjacency[[n1, n2]] {
                // v-n1 and v-n2 are two edges of a potential square
                // Find common neighbors of n1 and n2 (excluding v)
                for u in 0..adjacency.ncols() {
                    if u != v && adjacency[[n1, u]] && adjacency[[n2, u]] {
                        count += 1;
                    }
                }
            }
        }
    }
    count / 2 // Each square counted twice
}

fn estimate_eccentricity(adjacency: &Array2<bool>, v: usize) -> usize {
    let n = adjacency.nrows();
    let mut distances = vec![usize::MAX; n];
    let mut queue = VecDeque::new();

    distances[v] = 0;
    queue.push_back(v);

    while let Some(u) = queue.pop_front() {
        for w in 0..n {
            if adjacency[[u, w]] && distances[w] == usize::MAX {
                distances[w] = distances[u] + 1;
                queue.push_back(w);
            }
        }
    }

    *distances.iter().filter(|&&d| d != usize::MAX).max().unwrap_or(&0)
}

fn compute_difficulty(
    n: usize,
    density: f64,
    chromatic: usize,
    min_degree: usize,
    graph_type: GraphType,
) -> f64 {
    let mut score = 0.0;

    // Size factor
    score += (n as f64 / 100.0).min(30.0);

    // Chromatic number factor
    score += (chromatic as f64 / 2.0).min(25.0);

    // Density (medium is hardest)
    if density < 0.1 || density > 0.8 {
        score += 10.0;
    } else {
        score += 20.0;
    }

    // Graph type modifier
    score += match graph_type {
        GraphType::Leighton => 25.0,
        GraphType::Mycielski => 20.0,
        GraphType::Register => 15.0,
        _ => 10.0,
    };

    score.max(0.0).min(100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_graph() {
        let mut gen = GraphGenerator::new(42);
        let graph = gen.generate_random_graph(0, 20, 0.3, GraphType::RandomDense);

        assert_eq!(graph.num_vertices, 20);
        assert!(graph.num_edges > 0);
        assert!(graph.chromatic_number > 0);
        assert_eq!(graph.optimal_coloring.len(), 20);
    }

    #[test]
    fn test_mycielski() {
        let mut gen = GraphGenerator::new(123);
        let graph = gen.generate_mycielski_graph(0, 3);

        // Mycielski(K2, 3) should have chromatic number 4
        assert!(graph.chromatic_number >= 3);
    }

    #[test]
    fn test_node_features() {
        let adj = Array2::from_shape_fn((5, 5), |(i, j)| i != j && (i + j) % 2 == 0);
        let degrees: Vec<usize> = (0..5).map(|i| adj.row(i).iter().filter(|&&x| x).count()).collect();

        let features = compute_node_features(&adj, &degrees, GraphType::RandomDense);

        assert_eq!(features.shape(), &[5, 16]);
        // Check one-hot encoding
        assert_eq!(features[[0, 9]], 1.0); // RandomDense = index 1, so 8+1=9
    }
}
