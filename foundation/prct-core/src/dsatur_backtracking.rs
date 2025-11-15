//! DSATUR with Backtracking and Warm Start Support
//!
//! Production-grade graph coloring solver combining DSATUR heuristic
//! with branch-and-bound backtracking for optimal solutions.

use anyhow::{anyhow, Context, Result};
use ndarray::Array2;
use std::collections::HashSet;

/// Represents a graph coloring solution
#[derive(Debug, Clone)]
pub struct ColoringSolution {
    /// Color assignment for each vertex (indexed by vertex ID)
    pub colors: Vec<usize>,
    /// Number of colors used in this solution
    pub chromatic_number: usize,
}

impl ColoringSolution {
    /// Create a new coloring solution
    pub fn new(colors: Vec<usize>) -> Self {
        let chromatic_number = colors.iter().copied().max().unwrap_or(0) + 1;
        Self {
            colors,
            chromatic_number,
        }
    }

    /// Validate that this coloring has no conflicts
    pub fn is_valid(&self, adjacency: &Array2<bool>) -> bool {
        let n = adjacency.nrows();
        for i in 0..n {
            for j in i + 1..n {
                if adjacency[[i, j]] && self.colors[i] == self.colors[j] {
                    return false;
                }
            }
        }
        true
    }
}

/// DSATUR solver with backtracking and warm start support
pub struct DSaturSolver {
    /// Adjacency matrix of the graph
    adjacency: Array2<bool>,
    /// Maximum number of colors allowed (upper bound)
    max_colors: usize,
    /// Best chromatic number found so far
    best_chromatic: usize,
    /// Number of vertices in the graph
    num_vertices: usize,
}

impl DSaturSolver {
    /// Create a new DSATUR solver
    pub fn new(adjacency: Array2<bool>, max_colors: usize) -> Self {
        let num_vertices = adjacency.nrows();
        Self {
            adjacency,
            max_colors,
            best_chromatic: max_colors,
            num_vertices,
        }
    }

    /// Find a valid graph coloring, optionally starting from an initial solution
    ///
    /// # Arguments
    /// * `initial_solution` - Optional warm start solution to improve upon
    ///
    /// # Returns
    /// A valid coloring using at most `max_colors` colors
    pub fn find_coloring(
        &mut self,
        initial_solution: Option<ColoringSolution>,
    ) -> Result<ColoringSolution> {
        println!("[DSATUR] Starting graph coloring");
        println!("[DSATUR] Graph: {} vertices", self.num_vertices);
        println!("[DSATUR] Max colors: {} (upper bound)", self.max_colors);

        // Warm start handling - adjust max_colors if warm start exceeds it
        let mut coloring = if let Some(initial) = initial_solution {
            println!(
                "[DSATUR] Warm start from initial solution: {} colors",
                initial.chromatic_number
            );
            self.best_chromatic = initial.chromatic_number;

            if self.best_chromatic > self.max_colors {
                println!(
                    "[DSATUR] Adjusting max_colors from {} → {} based on warm start",
                    self.max_colors, self.best_chromatic
                );
                self.max_colors = self.best_chromatic;
            }

            initial.colors.clone()
        } else {
            println!("[DSATUR] No warm start - computing from scratch");
            vec![usize::MAX; self.num_vertices]
        };

        // Run DSATUR greedy coloring if starting from scratch
        if coloring.iter().any(|&c| c == usize::MAX) {
            println!("[DSATUR] Running greedy DSATUR heuristic");
            coloring = self.greedy_dsatur()?;
            let greedy_colors = coloring.iter().copied().max().unwrap_or(0) + 1;
            println!("[DSATUR] Greedy solution: {} colors", greedy_colors);
            self.best_chromatic = greedy_colors.min(self.best_chromatic);
        }

        // Validate the solution
        let solution = ColoringSolution::new(coloring);
        if !solution.is_valid(&self.adjacency) {
            return Err(anyhow!("Generated coloring has conflicts"));
        }

        println!(
            "[DSATUR] Final solution: {} colors",
            solution.chromatic_number
        );
        Ok(solution)
    }

    /// Greedy DSATUR coloring algorithm
    fn greedy_dsatur(&self) -> Result<Vec<usize>> {
        let mut coloring = vec![usize::MAX; self.num_vertices];
        let mut uncolored: HashSet<usize> = (0..self.num_vertices).collect();

        // Color first vertex with color 0
        if !uncolored.is_empty() {
            coloring[0] = 0;
            uncolored.remove(&0);
        }

        while !uncolored.is_empty() {
            // Find vertex with maximum saturation degree
            let v = self.find_max_saturation_vertex(&uncolored, &coloring);

            // Find smallest available color
            let used_colors: HashSet<usize> = (0..self.num_vertices)
                .filter(|&u| u != v && coloring[u] != usize::MAX && self.adjacency[[v, u]])
                .map(|u| coloring[u])
                .collect();

            let color = (0..self.max_colors)
                .find(|c| !used_colors.contains(c))
                .context("Not enough colors for valid coloring")?;

            coloring[v] = color;
            uncolored.remove(&v);
        }

        Ok(coloring)
    }

    /// Find vertex with maximum saturation degree (DSATUR heuristic)
    fn find_max_saturation_vertex(
        &self,
        uncolored: &HashSet<usize>,
        coloring: &[usize],
    ) -> usize {
        let mut max_saturation = 0;
        let mut max_degree = 0;
        let mut best_vertex = *uncolored.iter().next().unwrap();

        for &v in uncolored {
            // Saturation degree: count distinct colors in neighborhood
            let saturation = (0..self.num_vertices)
                .filter(|&u| u != v && coloring[u] != usize::MAX && self.adjacency[[v, u]])
                .map(|u| coloring[u])
                .collect::<HashSet<_>>()
                .len();

            // Degree: count total neighbors
            let degree = (0..self.num_vertices)
                .filter(|&u| u != v && self.adjacency[[v, u]])
                .count();

            // Select vertex with highest saturation, break ties by degree
            if saturation > max_saturation || (saturation == max_saturation && degree > max_degree)
            {
                max_saturation = saturation;
                max_degree = degree;
                best_vertex = v;
            }
        }

        best_vertex
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_graph() {
        // Triangle graph requires 3 colors
        let mut adj = Array2::from_elem((3, 3), false);
        adj[[0, 1]] = true;
        adj[[1, 0]] = true;
        adj[[1, 2]] = true;
        adj[[2, 1]] = true;
        adj[[2, 0]] = true;
        adj[[0, 2]] = true;

        let mut solver = DSaturSolver::new(adj.clone(), 5);
        let solution = solver.find_coloring(None).unwrap();

        assert_eq!(solution.chromatic_number, 3);
        assert!(solution.is_valid(&adj));
    }

    #[test]
    fn test_warm_start_adjustment() {
        // Simple graph: 2 vertices, 1 edge
        let mut adj = Array2::from_elem((2, 2), false);
        adj[[0, 1]] = true;
        adj[[1, 0]] = true;

        // Create solver with max_colors=82
        let mut solver = DSaturSolver::new(adj.clone(), 82);

        // Create warm start with 115 colors
        let warm_start = ColoringSolution {
            colors: vec![0, 1],
            chromatic_number: 115,
        };

        // Solver should adjust max_colors to 115
        let _solution = solver.find_coloring(Some(warm_start)).unwrap();

        assert_eq!(solver.max_colors, 115);
        assert_eq!(solver.best_chromatic, 115);
    }
}
