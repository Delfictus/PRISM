use prism_ai::{PrismAI, PrismConfig};

fn main() {
    env_logger::init();
    println!("PRISM-AI Demo (CPU fallback)\n==============================\n");

    // Simple 5-vertex cycle C5 (needs 3 colors)
    let n = 5;
    let mut adjacency = vec![vec![]; n];
    for i in 0..n {
        let j = (i + 1) % n;
        adjacency[i].push(j);
        adjacency[j].push(i);
    }

    let mut config = PrismConfig::default();
    config.use_gpu = false;

    let prism = PrismAI::new(config).expect("failed to init prism");
    let colors = prism.color_graph(adjacency).expect("coloring failed");

    let used = colors.iter().copied().max().unwrap_or(0) + 1;
    println!("Colored C{} using {} colors: {:?}", n, used, colors);
}
