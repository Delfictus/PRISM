// PRISM Configuration Management CLI
// Complete control over all parameters with runtime verification

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use toml::Value as TomlValue;

#[derive(Parser)]
#[clap(name = "prism-config")]
#[clap(about = "PRISM Configuration Management CLI", long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List all available parameters
    List {
        /// Filter by category (gpu, thermo, quantum, etc.)
        #[clap(short, long)]
        category: Option<String>,

        /// Show only modified parameters
        #[clap(short, long)]
        modified: bool,

        /// Show parameter types
        #[clap(short, long)]
        types: bool,
    },

    /// Get current value of a parameter
    Get {
        /// Parameter path (e.g., thermo.replicas)
        path: String,

        /// Config file to read from
        #[clap(short, long)]
        config: Option<PathBuf>,

        /// Show metadata
        #[clap(short, long)]
        verbose: bool,
    },

    /// Set parameter value
    Set {
        /// Parameter path (e.g., thermo.replicas)
        path: String,

        /// New value
        value: String,

        /// Config file to update
        #[clap(short, long, default_value = "config.toml")]
        config: PathBuf,

        /// Validate but don't apply
        #[clap(short, long)]
        dry_run: bool,
    },

    /// Apply a configuration file
    Apply {
        /// Path to TOML config file
        file: PathBuf,

        /// Target config to merge into
        #[clap(short, long)]
        target: Option<PathBuf>,

        /// Show what would change
        #[clap(short, long)]
        preview: bool,
    },

    /// Generate configuration file
    Generate {
        /// Output file path
        output: PathBuf,

        /// Base config to start from
        #[clap(short, long)]
        base: Option<PathBuf>,

        /// Template type (full, minimal, world-record)
        #[clap(short, long, default_value = "full")]
        template: String,
    },

    /// Validate configuration
    Validate {
        /// Config file to validate
        config: PathBuf,

        /// Check GPU memory requirements
        #[clap(short, long)]
        gpu: bool,

        /// Run deep validation
        #[clap(short, long)]
        deep: bool,
    },

    /// Show differences between configs
    Diff {
        /// First config file
        file1: PathBuf,

        /// Second config file
        file2: PathBuf,

        /// Show only changed values
        #[clap(short, long)]
        changes_only: bool,
    },

    /// Merge multiple configuration files
    Merge {
        /// Output file
        output: PathBuf,

        /// Base config file
        base: PathBuf,

        /// Layer files to merge (in order)
        #[clap(required = true)]
        layers: Vec<PathBuf>,
    },

    /// Tune parameters interactively
    Tune {
        /// Config file to tune
        config: PathBuf,

        /// Category to tune
        #[clap(short = 'c', long)]
        category: Option<String>,
    },

    /// Reset parameters to defaults
    Reset {
        /// Config file to reset
        config: PathBuf,

        /// Category to reset
        #[clap(short, long)]
        category: Option<String>,

        /// Output file (defaults to input)
        #[clap(short, long)]
        output: Option<PathBuf>,
    },
}

// Parameter metadata for validation and documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParameterMetadata {
    path: String,
    value_type: String,
    default: JsonValue,
    min: Option<JsonValue>,
    max: Option<JsonValue>,
    description: String,
    category: String,
    affects_gpu: bool,
    requires_restart: bool,
}

// Parameter bounds and defaults
fn get_parameter_metadata() -> HashMap<String, ParameterMetadata> {
    let mut metadata = HashMap::new();

    // GPU parameters
    metadata.insert(
        "gpu.device_id".to_string(),
        ParameterMetadata {
            path: "gpu.device_id".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(0),
            min: Some(JsonValue::from(0)),
            max: Some(JsonValue::from(8)),
            description: "CUDA device ID".to_string(),
            category: "gpu".to_string(),
            affects_gpu: true,
            requires_restart: true,
        },
    );

    metadata.insert(
        "gpu.batch_size".to_string(),
        ParameterMetadata {
            path: "gpu.batch_size".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(1024),
            min: Some(JsonValue::from(32)),
            max: Some(JsonValue::from(8192)),
            description: "GPU batch size for parallel operations".to_string(),
            category: "gpu".to_string(),
            affects_gpu: true,
            requires_restart: false,
        },
    );

    metadata.insert(
        "gpu.streams".to_string(),
        ParameterMetadata {
            path: "gpu.streams".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(4),
            min: Some(JsonValue::from(1)),
            max: Some(JsonValue::from(32)),
            description: "Number of CUDA streams".to_string(),
            category: "gpu".to_string(),
            affects_gpu: true,
            requires_restart: true,
        },
    );

    // Thermodynamic parameters
    metadata.insert(
        "thermo.replicas".to_string(),
        ParameterMetadata {
            path: "thermo.replicas".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(56),
            min: Some(JsonValue::from(1)),
            max: Some(JsonValue::from(56)), // VRAM limit
            description: "Number of temperature replicas (VRAM limited to 56 for 8GB)".to_string(),
            category: "thermo".to_string(),
            affects_gpu: true,
            requires_restart: false,
        },
    );

    metadata.insert(
        "thermo.num_temps".to_string(),
        ParameterMetadata {
            path: "thermo.num_temps".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(16),
            min: Some(JsonValue::from(2)),
            max: Some(JsonValue::from(56)),
            description: "Number of temperature levels".to_string(),
            category: "thermo".to_string(),
            affects_gpu: true,
            requires_restart: false,
        },
    );

    metadata.insert(
        "thermo.t_min".to_string(),
        ParameterMetadata {
            path: "thermo.t_min".to_string(),
            value_type: "f64".to_string(),
            default: JsonValue::from(0.01),
            min: Some(JsonValue::from(0.001)),
            max: Some(JsonValue::from(1.0)),
            description: "Minimum temperature".to_string(),
            category: "thermo".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    metadata.insert(
        "thermo.t_max".to_string(),
        ParameterMetadata {
            path: "thermo.t_max".to_string(),
            value_type: "f64".to_string(),
            default: JsonValue::from(10.0),
            min: Some(JsonValue::from(1.0)),
            max: Some(JsonValue::from(100.0)),
            description: "Maximum temperature".to_string(),
            category: "thermo".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    // Quantum parameters
    metadata.insert(
        "quantum.iterations".to_string(),
        ParameterMetadata {
            path: "quantum.iterations".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(20),
            min: Some(JsonValue::from(1)),
            max: Some(JsonValue::from(1000)),
            description: "Number of quantum solver iterations".to_string(),
            category: "quantum".to_string(),
            affects_gpu: true,
            requires_restart: false,
        },
    );

    metadata.insert(
        "quantum.target_chromatic".to_string(),
        ParameterMetadata {
            path: "quantum.target_chromatic".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(83),
            min: Some(JsonValue::from(1)),
            max: Some(JsonValue::from(10000)),
            description: "Target chromatic number".to_string(),
            category: "quantum".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    // Memetic parameters
    metadata.insert(
        "memetic.population_size".to_string(),
        ParameterMetadata {
            path: "memetic.population_size".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(256),
            min: Some(JsonValue::from(10)),
            max: Some(JsonValue::from(10000)),
            description: "Population size for memetic algorithm".to_string(),
            category: "memetic".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    metadata.insert(
        "memetic.generations".to_string(),
        ParameterMetadata {
            path: "memetic.generations".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(500),
            min: Some(JsonValue::from(1)),
            max: Some(JsonValue::from(100000)),
            description: "Number of generations".to_string(),
            category: "memetic".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    metadata.insert(
        "memetic.mutation_rate".to_string(),
        ParameterMetadata {
            path: "memetic.mutation_rate".to_string(),
            value_type: "f64".to_string(),
            default: JsonValue::from(0.15),
            min: Some(JsonValue::from(0.0)),
            max: Some(JsonValue::from(1.0)),
            description: "Mutation probability".to_string(),
            category: "memetic".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    // Phase toggles
    let phases = vec![
        (
            "use_reservoir_prediction",
            true,
            "Enable neuromorphic reservoir computing",
        ),
        (
            "use_active_inference",
            true,
            "Enable active inference optimization",
        ),
        (
            "use_transfer_entropy",
            true,
            "Enable transfer entropy analysis",
        ),
        (
            "use_thermodynamic_equilibration",
            true,
            "Enable thermodynamic equilibration",
        ),
        (
            "use_quantum_classical_hybrid",
            true,
            "Enable quantum-classical hybrid solver",
        ),
        (
            "use_multiscale_analysis",
            false,
            "Enable multi-scale analysis",
        ),
        (
            "use_ensemble_consensus",
            false,
            "Enable ensemble consensus voting",
        ),
        (
            "use_geodesic_features",
            false,
            "Enable geodesic feature extraction",
        ),
    ];

    for (name, default, desc) in phases {
        metadata.insert(
            name.to_string(),
            ParameterMetadata {
                path: name.to_string(),
                value_type: "bool".to_string(),
                default: JsonValue::from(default),
                min: None,
                max: None,
                description: desc.to_string(),
                category: "phases".to_string(),
                affects_gpu: name.contains("reservoir")
                    || name.contains("thermodynamic")
                    || name.contains("quantum"),
                requires_restart: false,
            },
        );
    }

    // Global parameters
    metadata.insert(
        "target_chromatic".to_string(),
        ParameterMetadata {
            path: "target_chromatic".to_string(),
            value_type: "usize".to_string(),
            default: JsonValue::from(83),
            min: Some(JsonValue::from(1)),
            max: Some(JsonValue::from(10000)),
            description: "Global target chromatic number".to_string(),
            category: "global".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    metadata.insert(
        "max_runtime_hours".to_string(),
        ParameterMetadata {
            path: "max_runtime_hours".to_string(),
            value_type: "f64".to_string(),
            default: JsonValue::from(48.0),
            min: Some(JsonValue::from(0.1)),
            max: Some(JsonValue::from(168.0)),
            description: "Maximum runtime in hours".to_string(),
            category: "global".to_string(),
            affects_gpu: false,
            requires_restart: false,
        },
    );

    metadata.insert(
        "deterministic".to_string(),
        ParameterMetadata {
            path: "deterministic".to_string(),
            value_type: "bool".to_string(),
            default: JsonValue::from(false),
            min: None,
            max: None,
            description: "Use deterministic random seed".to_string(),
            category: "global".to_string(),
            affects_gpu: false,
            requires_restart: true,
        },
    );

    metadata.insert(
        "seed".to_string(),
        ParameterMetadata {
            path: "seed".to_string(),
            value_type: "u64".to_string(),
            default: JsonValue::from(42),
            min: Some(JsonValue::from(0)),
            max: Some(JsonValue::from(u64::MAX)),
            description: "Random seed for deterministic mode".to_string(),
            category: "global".to_string(),
            affects_gpu: false,
            requires_restart: true,
        },
    );

    metadata
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::List {
            category,
            modified,
            types,
        } => {
            list_parameters(category, modified, types)?;
        }
        Commands::Get {
            path,
            config,
            verbose,
        } => {
            get_parameter(&path, config, verbose)?;
        }
        Commands::Set {
            path,
            value,
            config,
            dry_run,
        } => {
            set_parameter(&path, &value, &config, dry_run)?;
        }
        Commands::Apply {
            file,
            target,
            preview,
        } => {
            apply_config(&file, target, preview)?;
        }
        Commands::Generate {
            output,
            base,
            template,
        } => {
            generate_config(&output, base, &template)?;
        }
        Commands::Validate { config, gpu, deep } => {
            validate_config(&config, gpu, deep)?;
        }
        Commands::Diff {
            file1,
            file2,
            changes_only,
        } => {
            diff_configs(&file1, &file2, changes_only)?;
        }
        Commands::Merge {
            output,
            base,
            layers,
        } => {
            merge_configs(&output, &base, &layers)?;
        }
        Commands::Tune { config, category } => {
            tune_interactive(&config, category)?;
        }
        Commands::Reset {
            config,
            category,
            output,
        } => {
            reset_parameters(&config, category, output)?;
        }
    }

    Ok(())
}

fn list_parameters(category: Option<String>, _modified: bool, types: bool) -> Result<()> {
    let metadata = get_parameter_metadata();

    println!(
        "{}",
        "═══════════════════════════════════════════════════════════".blue()
    );
    println!(
        "{}",
        "                 PRISM CONFIGURATION PARAMETERS            "
            .blue()
            .bold()
    );
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════".blue()
    );

    // Group by category
    let mut categories: HashMap<String, Vec<&ParameterMetadata>> = HashMap::new();
    for param in metadata.values() {
        if let Some(ref cat) = category {
            if &param.category != cat {
                continue;
            }
        }
        categories
            .entry(param.category.clone())
            .or_insert_with(Vec::new)
            .push(param);
    }

    // Sort categories for consistent display
    let mut sorted_categories: Vec<_> = categories.into_iter().collect();
    sorted_categories.sort_by_key(|(cat, _)| cat.clone());

    for (cat, mut params) in sorted_categories {
        println!("\n{} {}", "►".cyan(), cat.to_uppercase().cyan().bold());
        println!("{}", "─".repeat(60).dimmed());

        // Sort parameters by path
        params.sort_by_key(|p| &p.path);

        for param in params {
            let gpu_indicator = if param.affects_gpu {
                "GPU".yellow()
            } else {
                "   ".normal()
            };

            let restart_indicator = if param.requires_restart {
                "®".red()
            } else {
                " ".normal()
            };

            if types {
                println!(
                    "  {} {} {:<35} [{:>8}] = {:>10}",
                    gpu_indicator,
                    restart_indicator,
                    param.path.white(),
                    param.value_type.dimmed(),
                    format_json_value(&param.default).yellow()
                );
            } else {
                println!(
                    "  {} {} {:<35} = {:>10}",
                    gpu_indicator,
                    restart_indicator,
                    param.path.white(),
                    format_json_value(&param.default).yellow()
                );
            }

            if param.min.is_some() || param.max.is_some() {
                let bounds = format!(
                    "      Range: [{} .. {}]",
                    param
                        .min
                        .as_ref()
                        .map(format_json_value)
                        .unwrap_or("∞".to_string()),
                    param
                        .max
                        .as_ref()
                        .map(format_json_value)
                        .unwrap_or("∞".to_string())
                );
                println!("{}", bounds.dimmed());
            }

            println!("      {}", param.description.dimmed());
        }
    }

    println!(
        "\n{}",
        "═══════════════════════════════════════════════════════════".blue()
    );
    println!(
        "Total Parameters: {} | GPU-Affecting: {} | Restart-Required: {}",
        metadata.len().to_string().white().bold(),
        metadata
            .values()
            .filter(|p| p.affects_gpu)
            .count()
            .to_string()
            .yellow()
            .bold(),
        metadata
            .values()
            .filter(|p| p.requires_restart)
            .count()
            .to_string()
            .red()
            .bold()
    );

    println!(
        "\nLegend: {} = Affects GPU | {} = Requires Restart",
        "GPU".yellow(),
        "®".red()
    );

    Ok(())
}

fn get_parameter(path: &str, config: Option<PathBuf>, verbose: bool) -> Result<()> {
    let metadata = get_parameter_metadata();

    // Load config if provided
    let value = if let Some(config_path) = config {
        let content = fs::read_to_string(&config_path)
            .context(format!("Failed to read config file: {:?}", config_path))?;
        let toml_value: TomlValue = toml::from_str(&content)?;
        get_toml_value(&toml_value, path)
    } else {
        metadata.get(path).map(|m| toml_value_from_json(&m.default))
    };

    if let Some(val) = value {
        if verbose {
            if let Some(meta) = metadata.get(path) {
                println!("{}", "╔════════════════════════════════════════╗".blue());
                println!(
                    "{} {} {}",
                    "║".blue(),
                    format!("Parameter: {}", path).white().bold(),
                    "║".blue()
                );
                println!("{}", "╚════════════════════════════════════════╝".blue());

                println!("  {} {}", "Type:".dimmed(), meta.value_type.yellow());
                println!(
                    "  {} {}",
                    "Current:".dimmed(),
                    format_toml_value(&val).green().bold()
                );
                println!(
                    "  {} {}",
                    "Default:".dimmed(),
                    format_json_value(&meta.default).white()
                );
                println!("  {} {}", "Category:".dimmed(), meta.category.cyan());
                println!("  {} {}", "Description:".dimmed(), meta.description);

                if meta.min.is_some() || meta.max.is_some() {
                    println!(
                        "  {} [{} .. {}]",
                        "Range:".dimmed(),
                        meta.min
                            .as_ref()
                            .map(format_json_value)
                            .unwrap_or("∞".to_string())
                            .yellow(),
                        meta.max
                            .as_ref()
                            .map(format_json_value)
                            .unwrap_or("∞".to_string())
                            .yellow()
                    );
                }

                println!(
                    "  {} {}",
                    "Affects GPU:".dimmed(),
                    if meta.affects_gpu {
                        "Yes".yellow()
                    } else {
                        "No".dimmed()
                    }
                );
                println!(
                    "  {} {}",
                    "Requires Restart:".dimmed(),
                    if meta.requires_restart {
                        "Yes".red()
                    } else {
                        "No".green()
                    }
                );
            }
        } else {
            println!("{}", format_toml_value(&val));
        }
    } else {
        eprintln!("{} Parameter not found: {}", "✗".red(), path);
        eprintln!("Use 'prism-config list' to see available parameters");
        std::process::exit(1);
    }

    Ok(())
}

fn set_parameter(path: &str, value: &str, config_path: &Path, dry_run: bool) -> Result<()> {
    let metadata = get_parameter_metadata();

    // Validate parameter exists
    let meta = metadata
        .get(path)
        .ok_or_else(|| anyhow!("Unknown parameter: {}", path))?;

    // Parse value
    let parsed_value = parse_toml_value(value, &meta.value_type)?;

    // Validate bounds
    if let Some(min) = &meta.min {
        if !validate_bound(&parsed_value, min, true)? {
            return Err(anyhow!(
                "{} below minimum: {} < {}",
                path,
                format_toml_value(&parsed_value),
                format_json_value(min)
            ));
        }
    }

    if let Some(max) = &meta.max {
        if !validate_bound(&parsed_value, max, false)? {
            return Err(anyhow!(
                "{} above maximum: {} > {}",
                path,
                format_toml_value(&parsed_value),
                format_json_value(max)
            ));
        }
    }

    if dry_run {
        println!("{} Dry run - validation passed", "►".cyan());
        println!(
            "{} Would set {} = {}",
            "✓".green(),
            path,
            format_toml_value(&parsed_value).yellow()
        );
        return Ok(());
    }

    // Load existing config or create new
    let mut toml_doc = if config_path.exists() {
        let content = fs::read_to_string(config_path)?;
        content.parse::<toml_edit::Document>()?
    } else {
        toml_edit::Document::new()
    };

    // Set the value
    set_toml_value(&mut toml_doc, path, parsed_value)?;

    // Write back
    fs::write(config_path, toml_doc.to_string())?;

    println!(
        "{} Set {} = {} in {}",
        "✓".green().bold(),
        path.white().bold(),
        value.yellow().bold(),
        config_path.display()
    );

    if meta.requires_restart {
        println!(
            "{} This parameter requires restarting the pipeline",
            "⚠".yellow()
        );
    }
    if meta.affects_gpu {
        println!("{} This parameter affects GPU operations", "⚡".yellow());
    }

    Ok(())
}

fn apply_config(file: &Path, target: Option<PathBuf>, preview: bool) -> Result<()> {
    println!("{} Loading config from {}", "►".cyan(), file.display());

    let content = fs::read_to_string(file)?;
    let source: TomlValue = toml::from_str(&content)?;

    let target_path = target.unwrap_or_else(|| PathBuf::from("config.toml"));

    if preview {
        println!("{} Preview mode - showing changes:", "►".cyan());

        if target_path.exists() {
            let target_content = fs::read_to_string(&target_path)?;
            let target_toml: TomlValue = toml::from_str(&target_content)?;
            show_diff(&target_toml, &source)?;
        } else {
            println!(
                "  Would create new config with all values from {}",
                file.display()
            );
        }

        println!("  (No changes applied - remove --preview to apply)");
    } else {
        fs::copy(file, &target_path)?;
        println!("{} Configuration applied successfully", "✓".green().bold());
        println!("  Source: {}", file.display());
        println!("  Target: {}", target_path.display());
    }

    Ok(())
}

fn generate_config(output: &Path, base: Option<PathBuf>, template: &str) -> Result<()> {
    println!("{} Generating {} config file...", "►".cyan(), template);

    let config = match template {
        "minimal" => generate_minimal_config(),
        "world-record" => generate_world_record_config(),
        _ => generate_full_config(),
    };

    // Merge with base if provided
    let final_config = if let Some(base_path) = base {
        let base_content = fs::read_to_string(&base_path)?;
        let base_toml: TomlValue = toml::from_str(&base_content)?;
        merge_toml(base_toml, config)?
    } else {
        config
    };

    let toml_str = toml::to_string_pretty(&final_config)?;
    fs::write(output, toml_str)?;

    println!(
        "{} Generated config file: {}",
        "✓".green().bold(),
        output.display()
    );

    Ok(())
}

fn validate_config(config_path: &Path, gpu: bool, deep: bool) -> Result<()> {
    println!("{} Validating configuration...", "►".cyan());

    let content = fs::read_to_string(config_path)?;
    let config: TomlValue = toml::from_str(&content)?;
    let metadata = get_parameter_metadata();

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Validate all parameters
    validate_toml_recursive(&config, "", &metadata, &mut errors, &mut warnings)?;

    if gpu {
        println!("  Checking GPU memory requirements...");

        // Get relevant parameters
        let replicas = get_toml_value(&config, "thermo.replicas")
            .and_then(|v| v.as_integer())
            .unwrap_or(56) as usize;

        let batch_size = get_toml_value(&config, "gpu.batch_size")
            .and_then(|v| v.as_integer())
            .unwrap_or(1024) as usize;

        // Estimate VRAM usage (simplified)
        let vram_mb = (replicas * batch_size * 8 * 2) / 1024 / 1024;

        if vram_mb > 8000 {
            errors.push(format!("VRAM usage {} MB exceeds 8GB limit", vram_mb));
        } else if vram_mb > 6000 {
            warnings.push(format!("VRAM usage {} MB is high (>75% of 8GB)", vram_mb));
        }

        println!("  Estimated VRAM: {} MB", vram_mb);
    }

    if deep {
        println!("  Running deep validation...");

        // Check phase dependencies
        let use_thermo = get_toml_value(&config, "use_thermodynamic_equilibration")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let gpu_thermo = get_toml_value(&config, "gpu.enable_thermo_gpu")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        if use_thermo && !gpu_thermo {
            warnings.push("Thermodynamic enabled but GPU acceleration disabled".to_string());
        }

        // Check for incompatible settings
        let deterministic = get_toml_value(&config, "deterministic")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let seed = get_toml_value(&config, "seed");

        if deterministic && seed.is_none() {
            warnings.push("Deterministic mode enabled but no seed specified".to_string());
        }
    }

    // Report results
    if errors.is_empty() && warnings.is_empty() {
        println!("{} Configuration valid!", "✓".green().bold());
    } else {
        if !errors.is_empty() {
            println!("\n{} Errors:", "✗".red().bold());
            for error in &errors {
                println!("  • {}", error.red());
            }
        }

        if !warnings.is_empty() {
            println!("\n{} Warnings:", "⚠".yellow());
            for warning in &warnings {
                println!("  • {}", warning.yellow());
            }
        }

        if !errors.is_empty() {
            std::process::exit(1);
        }
    }

    Ok(())
}

fn diff_configs(file1: &Path, file2: &Path, changes_only: bool) -> Result<()> {
    println!("{} Comparing configurations...", "►".cyan());

    let content1 = fs::read_to_string(file1)?;
    let content2 = fs::read_to_string(file2)?;

    let toml1: TomlValue = toml::from_str(&content1)?;
    let toml2: TomlValue = toml::from_str(&content2)?;

    println!(
        "\n{} {} → {}",
        "Diff:".cyan().bold(),
        file1.display().to_string().white(),
        file2.display().to_string().white()
    );
    println!("{}", "─".repeat(60).dimmed());

    show_diff_recursive(&toml1, &toml2, "", changes_only)?;

    Ok(())
}

fn merge_configs(output: &Path, base: &Path, layers: &[PathBuf]) -> Result<()> {
    println!("{} Merging configurations...", "►".cyan());

    let base_content = fs::read_to_string(base)?;
    let mut result: TomlValue = toml::from_str(&base_content)?;

    println!("  Base: {}", base.display());

    for layer_path in layers {
        println!("  Applying: {}", layer_path.display());
        let layer_content = fs::read_to_string(layer_path)?;
        let layer_toml: TomlValue = toml::from_str(&layer_content)?;
        result = merge_toml(result, layer_toml)?;
    }

    let toml_str = toml::to_string_pretty(&result)?;
    fs::write(output, toml_str)?;

    println!(
        "{} Merged config written to: {}",
        "✓".green().bold(),
        output.display()
    );

    Ok(())
}

fn tune_interactive(config_path: &Path, category: Option<String>) -> Result<()> {
    println!("{}", "╔════════════════════════════════════════╗".blue());
    println!(
        "{} {} {}",
        "║".blue(),
        "INTERACTIVE CONFIGURATION TUNING".white().bold(),
        "║".blue()
    );
    println!("{}", "╚════════════════════════════════════════╝".blue());

    let metadata = get_parameter_metadata();

    // Filter by category if specified
    let params_to_tune: Vec<_> = metadata
        .values()
        .filter(|p| category.as_ref().map_or(true, |c| &p.category == c))
        .collect();

    println!("\nTuning {} parameters", params_to_tune.len());

    // Load existing config
    let content = fs::read_to_string(config_path)?;
    let mut doc = content.parse::<toml_edit::Document>()?;

    // Also parse as TomlValue for reading values
    let toml_content: TomlValue = toml::from_str(&content)?;

    use std::io::{self, Write};

    for param in params_to_tune {
        // Get current value
        let current = get_toml_value(&toml_content, &param.path)
            .unwrap_or_else(|| toml_value_from_json(&param.default));

        println!("\n{} {}", "►".cyan(), param.path.white().bold());
        println!("  {}", param.description.dimmed());
        println!(
            "  Type: {} | Category: {}",
            param.value_type.yellow(),
            param.category.cyan()
        );

        if param.min.is_some() || param.max.is_some() {
            println!(
                "  Range: [{} .. {}]",
                param
                    .min
                    .as_ref()
                    .map(format_json_value)
                    .unwrap_or("∞".to_string()),
                param
                    .max
                    .as_ref()
                    .map(format_json_value)
                    .unwrap_or("∞".to_string())
            );
        }

        println!("  Current: {}", format_toml_value(&current).green());
        print!("  New value (or Enter to skip): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if !input.is_empty() {
            match parse_toml_value(input, &param.value_type) {
                Ok(new_value) => {
                    // Validate
                    let mut valid = true;
                    if let Some(min) = &param.min {
                        if !validate_bound(&new_value, min, true)? {
                            println!("  {} Value below minimum", "✗".red());
                            valid = false;
                        }
                    }
                    if valid && param.max.is_some() {
                        if let Some(max) = &param.max {
                            if !validate_bound(&new_value, max, false)? {
                                println!("  {} Value above maximum", "✗".red());
                                valid = false;
                            }
                        }
                    }

                    if valid {
                        set_toml_value(&mut doc, &param.path, new_value)?;
                        println!("  {} Updated to {}", "✓".green(), input.yellow());
                    }
                }
                Err(e) => {
                    println!("  {} Invalid value: {}", "✗".red(), e);
                }
            }
        }
    }

    // Save
    fs::write(config_path, doc.to_string())?;
    println!(
        "\n{} Configuration saved to {}",
        "✓".green().bold(),
        config_path.display()
    );

    Ok(())
}

fn reset_parameters(
    config_path: &Path,
    category: Option<String>,
    output: Option<PathBuf>,
) -> Result<()> {
    let metadata = get_parameter_metadata();
    let output_path = output.unwrap_or_else(|| config_path.to_path_buf());

    let mut doc = if config_path.exists() {
        let content = fs::read_to_string(config_path)?;
        content.parse::<toml_edit::Document>()?
    } else {
        toml_edit::Document::new()
    };

    let mut reset_count = 0;

    for (path, meta) in &metadata {
        if category.as_ref().map_or(false, |c| &meta.category != c) {
            continue;
        }

        let default = toml_value_from_json(&meta.default);
        set_toml_value(&mut doc, path, default)?;
        reset_count += 1;
    }

    fs::write(&output_path, doc.to_string())?;

    println!(
        "{} Reset {} parameters to defaults in {}",
        "✓".green().bold(),
        reset_count,
        output_path.display()
    );

    Ok(())
}

// Helper functions

fn format_json_value(value: &JsonValue) -> String {
    match value {
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => n.to_string(),
        JsonValue::String(s) => format!("\"{}\"", s),
        _ => value.to_string(),
    }
}

fn format_toml_value(value: &TomlValue) -> String {
    match value {
        TomlValue::Boolean(b) => b.to_string(),
        TomlValue::Integer(i) => i.to_string(),
        TomlValue::Float(f) => f.to_string(),
        TomlValue::String(s) => format!("\"{}\"", s),
        _ => value.to_string(),
    }
}

fn parse_toml_value(s: &str, expected_type: &str) -> Result<TomlValue> {
    match expected_type {
        "bool" => {
            let b = s.parse::<bool>().context("Expected boolean")?;
            Ok(TomlValue::Boolean(b))
        }
        "usize" | "u32" | "u64" | "i64" => {
            let i = s.parse::<i64>().context("Expected integer")?;
            Ok(TomlValue::Integer(i))
        }
        "f32" | "f64" => {
            let f = s.parse::<f64>().context("Expected float")?;
            Ok(TomlValue::Float(f))
        }
        _ => Ok(TomlValue::String(s.to_string())),
    }
}

fn toml_value_from_json(json: &JsonValue) -> TomlValue {
    match json {
        JsonValue::Bool(b) => TomlValue::Boolean(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                TomlValue::Integer(i)
            } else if let Some(f) = n.as_f64() {
                TomlValue::Float(f)
            } else {
                TomlValue::String(n.to_string())
            }
        }
        JsonValue::String(s) => TomlValue::String(s.clone()),
        _ => TomlValue::String(json.to_string()),
    }
}

fn get_toml_value(toml: &TomlValue, path: &str) -> Option<TomlValue> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = toml;

    for part in parts {
        match current {
            TomlValue::Table(table) => {
                current = table.get(part)?;
            }
            _ => return None,
        }
    }

    Some(current.clone())
}

fn set_toml_value(doc: &mut toml_edit::Document, path: &str, value: TomlValue) -> Result<()> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = doc.as_table_mut();

    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            // Set the final value
            current[part] = toml_edit::value(toml_value_to_edit(value.clone()));
        } else {
            // Create nested table if needed
            if !current.contains_key(part) || !current[part].is_table() {
                current[part] = toml_edit::table();
            }
            current = current[part]
                .as_table_mut()
                .ok_or_else(|| anyhow!("Failed to create nested table"))?;
        }
    }

    Ok(())
}

fn toml_value_to_edit(value: TomlValue) -> toml_edit::Value {
    match value {
        TomlValue::Boolean(b) => toml_edit::Value::from(b),
        TomlValue::Integer(i) => toml_edit::Value::from(i),
        TomlValue::Float(f) => toml_edit::Value::from(f),
        TomlValue::String(s) => toml_edit::Value::from(s),
        _ => toml_edit::Value::from(value.to_string()),
    }
}

fn validate_bound(value: &TomlValue, bound: &JsonValue, is_min: bool) -> Result<bool> {
    let value_num = match value {
        TomlValue::Integer(i) => *i as f64,
        TomlValue::Float(f) => *f,
        _ => return Ok(true),
    };

    let bound_num = bound
        .as_f64()
        .or_else(|| bound.as_i64().map(|i| i as f64))
        .or_else(|| bound.as_u64().map(|u| u as f64))
        .ok_or_else(|| anyhow!("Invalid bound type"))?;

    Ok(if is_min {
        value_num >= bound_num
    } else {
        value_num <= bound_num
    })
}

fn validate_toml_recursive(
    toml: &TomlValue,
    prefix: &str,
    metadata: &HashMap<String, ParameterMetadata>,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    match toml {
        TomlValue::Table(table) => {
            for (key, value) in table {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };

                if let Some(meta) = metadata.get(&path) {
                    // Validate type and bounds
                    if let Some(min) = &meta.min {
                        if !validate_bound(value, min, true)? {
                            errors.push(format!("{} below minimum", path));
                        }
                    }

                    if let Some(max) = &meta.max {
                        if !validate_bound(value, max, false)? {
                            errors.push(format!("{} above maximum", path));
                        }
                    }
                } else if !value.is_table() {
                    warnings.push(format!("Unknown parameter: {}", path));
                }

                validate_toml_recursive(value, &path, metadata, errors, warnings)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn show_diff(old: &TomlValue, new: &TomlValue) -> Result<()> {
    show_diff_recursive(old, new, "", false)
}

fn show_diff_recursive(
    old: &TomlValue,
    new: &TomlValue,
    prefix: &str,
    changes_only: bool,
) -> Result<()> {
    match (old, new) {
        (TomlValue::Table(old_table), TomlValue::Table(new_table)) => {
            let mut all_keys = BTreeMap::new();
            for key in old_table.keys() {
                all_keys.insert(key.clone(), true);
            }
            for key in new_table.keys() {
                all_keys.insert(key.clone(), true);
            }

            for key in all_keys.keys() {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };

                match (old_table.get(key), new_table.get(key)) {
                    (Some(old_val), Some(new_val)) => {
                        if old_val != new_val {
                            if !old_val.is_table() && !new_val.is_table() {
                                println!(
                                    "  {} {} → {}",
                                    path.yellow(),
                                    format_toml_value(old_val).red(),
                                    format_toml_value(new_val).green()
                                );
                            } else {
                                show_diff_recursive(old_val, new_val, &path, changes_only)?;
                            }
                        } else if !changes_only && !old_val.is_table() {
                            println!(
                                "  {} {}",
                                path.dimmed(),
                                format_toml_value(old_val).dimmed()
                            );
                        }
                    }
                    (Some(old_val), None) => {
                        println!(
                            "  {} {} (removed)",
                            path.red(),
                            format_toml_value(old_val).red()
                        );
                    }
                    (None, Some(new_val)) => {
                        println!(
                            "  {} {} (added)",
                            path.green(),
                            format_toml_value(new_val).green()
                        );
                    }
                    _ => {}
                }
            }
        }
        _ => {
            if old != new {
                println!(
                    "  {} {} → {}",
                    prefix.yellow(),
                    format_toml_value(old).red(),
                    format_toml_value(new).green()
                );
            }
        }
    }

    Ok(())
}

fn merge_toml(base: TomlValue, overlay: TomlValue) -> Result<TomlValue> {
    match (base, overlay) {
        (TomlValue::Table(mut base_table), TomlValue::Table(overlay_table)) => {
            for (key, value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) if base_value.is_table() && value.is_table() => {
                        *base_value = merge_toml(base_value.clone(), value)?;
                    }
                    _ => {
                        base_table.insert(key, value);
                    }
                }
            }
            Ok(TomlValue::Table(base_table))
        }
        (_, overlay) => Ok(overlay),
    }
}

fn generate_minimal_config() -> TomlValue {
    let value = toml::toml! {
        target_chromatic = 83
        max_runtime_hours = 1.0
        deterministic = true
        seed = 42

        [gpu]
        device_id = 0
        batch_size = 1024

        [thermo]
        replicas = 32
        num_temps = 16
    };
    TomlValue::Table(value)
}

fn generate_world_record_config() -> TomlValue {
    let value = toml::toml! {
        profile = "world_record"
        version = "1.1.0"
        target_chromatic = 83
        max_runtime_hours = 48.0
        deterministic = false

        use_reservoir_prediction = true
        use_active_inference = true
        use_transfer_entropy = true
        use_thermodynamic_equilibration = true
        use_quantum_classical_hybrid = true
        use_multiscale_analysis = true
        use_ensemble_consensus = true
        use_geodesic_features = true

        [gpu]
        device_id = 0
        streams = 8
        batch_size = 2048
        enable_reservoir_gpu = true
        enable_te_gpu = true
        enable_statmech_gpu = true
        enable_thermo_gpu = true
        enable_quantum_gpu = true

        [thermo]
        replicas = 56
        num_temps = 56
        t_min = 0.001
        t_max = 20.0
        exchange_interval = 100
        steps_per_temp = 10000

        [quantum]
        iterations = 50
        target_chromatic = 83
        failure_retries = 5
        fallback_on_failure = true

        [memetic]
        population_size = 512
        elite_size = 16
        generations = 10000
        mutation_rate = 0.05
        local_search_depth = 50000
    };
    TomlValue::Table(value)
}

fn generate_full_config() -> TomlValue {
    let metadata = get_parameter_metadata();
    let mut config = toml::map::Map::new();

    for (path, meta) in metadata {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = &mut config;

        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                current.insert(part.to_string(), toml_value_from_json(&meta.default));
            } else {
                current = current
                    .entry(part.to_string())
                    .or_insert_with(|| TomlValue::Table(toml::map::Map::new()))
                    .as_table_mut()
                    .unwrap();
            }
        }
    }

    TomlValue::Table(config)
}
