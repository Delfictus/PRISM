//! Data loading and graph generation module

mod dimacs_parser;

pub use dimacs_parser::{DensityClass, DimacsGraph, GraphCharacteristics, GraphType, StrategyMix};

use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub struct DIMACParser;

impl DIMACParser {
    pub fn parse_file(path: &Path) -> Result<Vec<Vec<usize>>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut n = 0;
        let mut adjacency = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.starts_with('c') {
                // Comment line
                continue;
            } else if line.starts_with('p') {
                // Problem line: p edge n m
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    n = parts[2].parse::<usize>().unwrap_or(0);
                    adjacency = vec![Vec::new(); n];
                }
            } else if line.starts_with('e') {
                // Edge line: e v1 v2
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    if let (Ok(v1), Ok(v2)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                    {
                        if v1 > 0 && v2 > 0 && v1 <= n && v2 <= n {
                            adjacency[v1 - 1].push(v2 - 1);
                            adjacency[v2 - 1].push(v1 - 1);
                        }
                    }
                }
            }
        }

        Ok(adjacency)
    }
}

pub struct GraphGenerator;

impl GraphGenerator {
    pub fn complete_graph(n: usize) -> Vec<Vec<usize>> {
        let mut adjacency = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adjacency[i].push(j);
                }
            }
        }
        adjacency
    }

    pub fn random_graph(n: usize, p: f64) -> Vec<Vec<usize>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut adjacency = vec![Vec::new(); n];

        for i in 0..n {
            for j in i + 1..n {
                if rng.gen::<f64>() < p {
                    adjacency[i].push(j);
                    adjacency[j].push(i);
                }
            }
        }
        adjacency
    }
}
