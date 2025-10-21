fn main() {
    let registry = prism_ai::features::registry();
    let manifest = registry.snapshot();
    println!("Meta registry Merkle root: {}", manifest.merkle_root);
}
