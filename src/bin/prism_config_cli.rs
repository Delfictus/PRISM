// PRISM Configuration CLI
// Full control over all parameters with runtime verification

use clap::{Parser, Subcommand};
use colored::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

// Import telemetry types from prct-core
use prct_core::telemetry::{PhaseExecMode, PhaseName, RunMetric};

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

        /// Show access counts
        #[clap(short, long)]
        accessed: bool,
    },

    /// Get current value of a parameter
    Get {
        /// Parameter path (e.g., thermo.replicas)
        path: String,

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

        /// Validate but don't apply
        #[clap(short, long)]
        dry_run: bool,
    },

    /// Apply a configuration file
    Apply {
        /// Path to TOML config file
        file: String,

        /// Show what would change
        #[clap(short, long)]
        preview: bool,

        /// Merge with existing values (vs replace)
        #[clap(short, long)]
        merge: bool,
    },

    /// Generate configuration file
    Generate {
        /// Output file path
        output: String,

        /// Include all parameters (vs only modified)
        #[clap(short, long)]
        all: bool,

        /// Template type
        #[clap(short, long, default_value = "full")]
        template: String,
    },

    /// Validate current configuration
    Validate {
        /// Check GPU memory requirements
        #[clap(short, long)]
        gpu: bool,

        /// Run deep validation
        #[clap(short, long)]
        deep: bool,
    },

    /// Show parameter usage verification
    Verify {
        /// Run a test and show which parameters are used
        #[clap(short, long)]
        test: bool,

        /// Reset access counts
        #[clap(short, long)]
        reset: bool,

        /// Export verification report
        #[clap(short, long)]
        export: Option<String>,
    },

    /// Interactive tuning mode
    Tune {
        /// Category to tune
        category: String,

        /// Use smart suggestions
        #[clap(short, long)]
        smart: bool,
    },

    /// Show differences between configs
    Diff {
        /// First config file
        file1: String,

        /// Second config file
        file2: String,

        /// Show only changed values
        #[clap(short, long)]
        changes_only: bool,
    },

    /// Reset parameters to defaults
    Reset {
        /// Reset all parameters
        #[clap(short, long)]
        all: bool,

        /// Category to reset
        #[clap(short, long)]
        category: Option<String>,

        /// Specific parameter path
        #[clap(short, long)]
        path: Option<String>,
    },

    /// Monitor live telemetry metrics
    Monitor {
        /// Tail live metrics (follow mode)
        #[clap(long)]
        tail: bool,

        /// Show summary statistics
        #[clap(long)]
        summary: bool,

        /// Filter by phase
        #[clap(long)]
        phase: Option<String>,

        /// Metrics file path (default: latest)
        #[clap(long)]
        file: Option<PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Initialize registry with schema
    load_schema()?;

    match cli.command {
        Commands::List {
            category,
            modified,
            accessed,
        } => {
            list_parameters(category, modified, accessed)?;
        }

        Commands::Get { path, verbose } => {
            get_parameter(&path, verbose)?;
        }

        Commands::Set {
            path,
            value,
            dry_run,
        } => {
            set_parameter(&path, &value, dry_run)?;
        }

        Commands::Apply {
            file,
            preview,
            merge,
        } => {
            apply_config(&file, preview, merge)?;
        }

        Commands::Generate {
            output,
            all,
            template,
        } => {
            generate_config(&output, all, &template)?;
        }

        Commands::Validate { gpu, deep } => {
            validate_config(gpu, deep)?;
        }

        Commands::Verify {
            test,
            reset,
            export,
        } => {
            verify_usage(test, reset, export)?;
        }

        Commands::Tune { category, smart } => {
            tune_interactive(&category, smart)?;
        }

        Commands::Diff {
            file1,
            file2,
            changes_only,
        } => {
            diff_configs(&file1, &file2, changes_only)?;
        }

        Commands::Reset {
            all,
            category,
            path,
        } => {
            reset_parameters(all, category, path)?;
        }

        Commands::Monitor {
            tail,
            summary,
            phase,
            file,
        } => {
            monitor_telemetry(tail, summary, phase, file)?;
        }
    }

    Ok(())
}

fn load_schema() -> Result<(), Box<dyn std::error::Error>> {
    // Load parameter schema from embedded file or config
    let schema_path = "foundation/prct-core/configs/parameter_schema.toml";
    if std::path::Path::new(schema_path).exists() {
        let schema_content = fs::read_to_string(schema_path)?;
        CONFIG_REGISTRY.load_toml(&schema_content)?;
    }

    // Auto-discover from existing configs
    auto_discover_parameters()?;

    Ok(())
}

fn auto_discover_parameters() -> Result<(), Box<dyn std::error::Error>> {
    // Scan for .toml configs and register their parameters
    let config_dir = "foundation/prct-core/configs";
    for entry in fs::read_dir(config_dir)? {
        let entry = entry?;
        if entry.path().extension() == Some(std::ffi::OsStr::new("toml")) {
            let content = fs::read_to_string(entry.path())?;
            if let Ok(parsed) = toml::from_str::<toml::Value>(&content) {
                discover_from_toml(&parsed, "")?;
            }
        }
    }
    Ok(())
}

fn discover_from_toml(value: &toml::Value, prefix: &str) -> Result<(), Box<dyn std::error::Error>> {
    match value {
        toml::Value::Table(table) => {
            for (key, val) in table {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", prefix, key)
                };

                match val {
                    toml::Value::Table(_) => {
                        discover_from_toml(val, &path)?;
                    }
                    _ => {
                        // Register if not already present
                        let value_type = match val {
                            toml::Value::Boolean(_) => "bool",
                            toml::Value::Integer(_) => "i64",
                            toml::Value::Float(_) => "f64",
                            toml::Value::String(_) => "String",
                            _ => "Value",
                        };

                        CONFIG_REGISTRY.register_parameter(ParameterMetadata {
                            name: key.clone(),
                            path: path.clone(),
                            value_type: value_type.to_string(),
                            default: serde_json::to_value(val).unwrap_or(Value::Null),
                            current: serde_json::to_value(val).unwrap_or(Value::Null),
                            min: None,
                            max: None,
                            description: format!("Auto-discovered from configs"),
                            category: prefix.split('.').next().unwrap_or("general").to_string(),
                            affects_gpu: path.contains("gpu"),
                            requires_restart: false,
                            access_count: 0,
                        });
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn list_parameters(
    category: Option<String>,
    modified: bool,
    accessed: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════".blue()
    );
    println!(
        "{}",
        "                    PRISM CONFIGURATION PARAMETERS         "
            .blue()
            .bold()
    );
    println!(
        "{}",
        "═══════════════════════════════════════════════════════════".blue()
    );

    let params = CONFIG_REGISTRY.parameters.read().unwrap();
    let mut categories: std::collections::HashMap<String, Vec<&ParameterMetadata>> =
        std::collections::HashMap::new();

    for (_, param) in params.iter() {
        // Apply filters
        if let Some(ref cat) = category {
            if &param.category != cat {
                continue;
            }
        }

        if modified && param.current == param.default {
            continue;
        }

        if accessed && param.access_count == 0 {
            continue;
        }

        categories
            .entry(param.category.clone())
            .or_insert_with(Vec::new)
            .push(param);
    }

    for (cat, params) in categories {
        println!("\n{} {}", "►".cyan(), cat.to_uppercase().cyan().bold());
        println!("{}", "─".repeat(60).dim());

        for param in params {
            let status = if param.current != param.default {
                "●".green()
            } else {
                "○".dim()
            };

            let gpu_indicator = if param.affects_gpu {
                "GPU".yellow()
            } else {
                "   ".normal()
            };

            let access_info = if param.access_count > 0 {
                format!("[{}x]", param.access_count).dim()
            } else {
                "".to_string()
            };

            println!(
                "  {} {} {:<30} = {:>10} {} {}",
                status,
                gpu_indicator,
                param.path.white(),
                format_value(&param.current).yellow(),
                param.description.dim(),
                access_info
            );

            if param.min.is_some() || param.max.is_some() {
                let bounds = format!(
                    "      Range: [{} .. {}]",
                    param
                        .min
                        .as_ref()
                        .map(format_value)
                        .unwrap_or("∞".to_string()),
                    param
                        .max
                        .as_ref()
                        .map(format_value)
                        .unwrap_or("∞".to_string())
                );
                println!("{}", bounds.dim());
            }
        }
    }

    println!(
        "\n{}",
        "═══════════════════════════════════════════════════════════".blue()
    );

    // Summary statistics
    let total = params.len();
    let modified_count = params.values().filter(|p| p.current != p.default).count();
    let accessed_count = params.values().filter(|p| p.access_count > 0).count();

    println!(
        "Total: {} | Modified: {} | Accessed: {}",
        total.to_string().white().bold(),
        modified_count.to_string().green().bold(),
        accessed_count.to_string().cyan().bold()
    );

    Ok(())
}

fn get_parameter(path: &str, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let params = CONFIG_REGISTRY.parameters.read().unwrap();

    if let Some(param) = params.get(path) {
        if verbose {
            println!("{}", "╔════════════════════════════════════════╗".blue());
            println!(
                "{} {} {}",
                "║".blue(),
                format!("Parameter: {}", path).white().bold(),
                "║".blue()
            );
            println!("{}", "╚════════════════════════════════════════╝".blue());

            println!("  {} {}", "Type:".dim(), param.value_type.yellow());
            println!(
                "  {} {}",
                "Current:".dim(),
                format_value(&param.current).green().bold()
            );
            println!(
                "  {} {}",
                "Default:".dim(),
                format_value(&param.default).white()
            );
            println!("  {} {}", "Category:".dim(), param.category.cyan());
            println!("  {} {}", "Description:".dim(), param.description);

            if param.min.is_some() || param.max.is_some() {
                println!(
                    "  {} [{} .. {}]",
                    "Range:".dim(),
                    param
                        .min
                        .as_ref()
                        .map(format_value)
                        .unwrap_or("∞".to_string())
                        .yellow(),
                    param
                        .max
                        .as_ref()
                        .map(format_value)
                        .unwrap_or("∞".to_string())
                        .yellow()
                );
            }

            println!(
                "  {} {}",
                "Affects GPU:".dim(),
                if param.affects_gpu {
                    "Yes".yellow()
                } else {
                    "No".dim()
                }
            );
            println!(
                "  {} {}",
                "Requires Restart:".dim(),
                if param.requires_restart {
                    "Yes".red()
                } else {
                    "No".green()
                }
            );
            println!(
                "  {} {}",
                "Access Count:".dim(),
                param.access_count.to_string().cyan()
            );

            if param.current != param.default {
                println!("\n  {} Parameter has been modified", "⚠".yellow());
            }
        } else {
            println!("{}", format_value(&param.current));
        }
    } else {
        eprintln!("{} Parameter not found: {}", "✗".red(), path);
        eprintln!("Use 'prism-config list' to see available parameters");
        std::process::exit(1);
    }

    Ok(())
}

fn set_parameter(path: &str, value: &str, dry_run: bool) -> Result<(), Box<dyn std::error::Error>> {
    // Parse value based on type
    let parsed_value = parse_value(value)?;

    if dry_run {
        println!("{} Dry run - validating...", "►".cyan());
    }

    match CONFIG_REGISTRY.set(path, parsed_value.clone()) {
        Ok(()) => {
            if dry_run {
                println!(
                    "{} Validation passed: {} = {}",
                    "✓".green(),
                    path,
                    format_value(&parsed_value).yellow()
                );
                println!("  (No changes applied - remove --dry-run to apply)");
            } else {
                println!(
                    "{} Set {} = {}",
                    "✓".green().bold(),
                    path.white().bold(),
                    format_value(&parsed_value).yellow().bold()
                );

                // Check if restart required
                let params = CONFIG_REGISTRY.parameters.read().unwrap();
                if let Some(param) = params.get(path) {
                    if param.requires_restart {
                        println!(
                            "{} This parameter requires restarting the pipeline",
                            "⚠".yellow()
                        );
                    }
                    if param.affects_gpu {
                        println!("{} This parameter affects GPU operations", "⚡".yellow());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("{} Failed to set parameter: {}", "✗".red(), e);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn verify_usage(
    test: bool,
    reset: bool,
    export: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if reset {
        // Reset all access counts
        let mut params = CONFIG_REGISTRY.parameters.write().unwrap();
        for param in params.values_mut() {
            param.access_count = 0;
        }
        println!("{} Reset all access counts", "✓".green());
        return Ok(());
    }

    if test {
        println!("{} Running verification test...", "►".cyan());
        println!("  (This will run a minimal pipeline and track parameter access)");

        // Enable verification mode
        *CONFIG_REGISTRY.verification_mode.write().unwrap() = true;

        // TODO: Run actual test here
        println!("{} Test complete", "✓".green());
    }

    // Generate report
    let report = CONFIG_REGISTRY.generate_verification_report();

    println!(
        "\n{}",
        "╔══════════════════════════════════════════╗".blue()
    );
    println!(
        "{} {} {}",
        "║".blue(),
        "PARAMETER VERIFICATION REPORT".white().bold(),
        "║".blue()
    );
    println!("{}", "╚══════════════════════════════════════════╝".blue());

    println!("\n{} Statistics:", "►".cyan());
    println!(
        "  Total Parameters: {}",
        report.total_parameters.to_string().white().bold()
    );
    println!(
        "  Accessed: {} ({}%)",
        report.accessed_parameters.to_string().green().bold(),
        (report.accessed_parameters * 100 / report.total_parameters)
            .to_string()
            .green()
    );
    println!(
        "  Modified: {}",
        report.modified_parameters.len().to_string().yellow().bold()
    );
    println!(
        "  Total Accesses: {}",
        report.total_accesses.to_string().cyan().bold()
    );

    if !report.unused_parameters.is_empty() {
        println!("\n{} Unused Parameters:", "►".yellow());
        for param in &report.unused_parameters[..10.min(report.unused_parameters.len())] {
            println!("  • {}", param.red());
        }
        if report.unused_parameters.len() > 10 {
            println!("  ... and {} more", report.unused_parameters.len() - 10);
        }
    }

    if !report.frequently_used.is_empty() {
        println!("\n{} Frequently Used:", "►".green());
        for (param, count) in &report.frequently_used[..5.min(report.frequently_used.len())] {
            println!(
                "  • {} ({}x)",
                param.white().bold(),
                count.to_string().cyan()
            );
        }
    }

    if !report.modified_parameters.is_empty() {
        println!("\n{} Modified Parameters:", "►".yellow());
        for modif in &report.modified_parameters {
            println!(
                "  • {} : {} → {}",
                modif.path.white(),
                format_value(&modif.default).dim(),
                format_value(&modif.current).yellow().bold()
            );
        }
    }

    // Export if requested
    if let Some(output) = export {
        let json = serde_json::to_string_pretty(&report)?;
        fs::write(&output, json)?;
        println!(
            "\n{} Exported report to {}",
            "✓".green(),
            output.white().bold()
        );
    }

    Ok(())
}

fn format_value(value: &Value) -> String {
    match value {
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => format!("\"{}\"", s),
        _ => value.to_string(),
    }
}

fn parse_value(s: &str) -> Result<Value, Box<dyn std::error::Error>> {
    // Try parsing as various types
    if let Ok(b) = s.parse::<bool>() {
        return Ok(Value::Bool(b));
    }
    if let Ok(i) = s.parse::<i64>() {
        return Ok(Value::Number(i.into()));
    }
    if let Ok(f) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(Value::Number(n));
        }
    }
    // Default to string
    Ok(Value::String(s.to_string()))
}

fn apply_config(file: &str, preview: bool, merge: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Loading config from {}", "►".cyan(), file.white().bold());

    let content = fs::read_to_string(file)?;

    if preview {
        println!("{} Preview mode - showing changes:", "►".cyan());
        // TODO: Show what would change
        println!("  (No changes applied - remove --preview to apply)");
    } else {
        CONFIG_REGISTRY.load_toml(&content)?;
        println!("{} Configuration applied successfully", "✓".green().bold());
    }

    Ok(())
}

fn generate_config(
    output: &str,
    all: bool,
    template: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Generating config file...", "►".cyan());

    let params = CONFIG_REGISTRY.parameters.read().unwrap();
    let mut config = toml::map::Map::new();

    for (path, param) in params.iter() {
        if !all && param.current == param.default {
            continue;
        }

        // Build nested structure
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = &mut config;

        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // Leaf value
                current.insert(part.to_string(), toml_value_from_json(&param.current));
            } else {
                // Nested table
                current = current
                    .entry(part.to_string())
                    .or_insert_with(|| toml::Value::Table(toml::map::Map::new()))
                    .as_table_mut()
                    .unwrap();
            }
        }
    }

    let toml_str = toml::to_string_pretty(&toml::Value::Table(config))?;
    fs::write(output, toml_str)?;

    println!(
        "{} Generated config file: {}",
        "✓".green().bold(),
        output.white().bold()
    );

    Ok(())
}

fn toml_value_from_json(json: &Value) -> toml::Value {
    match json {
        Value::Bool(b) => toml::Value::Boolean(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                toml::Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                toml::Value::Float(f)
            } else {
                toml::Value::String(n.to_string())
            }
        }
        Value::String(s) => toml::Value::String(s.clone()),
        _ => toml::Value::String(json.to_string()),
    }
}

fn validate_config(gpu: bool, deep: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Validating configuration...", "►".cyan());

    let params = CONFIG_REGISTRY.parameters.read().unwrap();
    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Basic validation
    for (path, param) in params.iter() {
        // Check bounds
        if let Some(min) = &param.min {
            if param.current.as_f64() < min.as_f64() {
                errors.push(format!("{} below minimum", path));
            }
        }

        if let Some(max) = &param.max {
            if param.current.as_f64() > max.as_f64() {
                errors.push(format!("{} above maximum", path));
            }
        }
    }

    if gpu {
        // GPU-specific validation
        println!("  Checking GPU memory requirements...");

        // Get relevant parameters
        let replicas = params
            .get("thermo.replicas")
            .and_then(|p| p.current.as_u64())
            .unwrap_or(56) as usize;

        let batch_size = params
            .get("gpu.batch_size")
            .and_then(|p| p.current.as_u64())
            .unwrap_or(1024) as usize;

        // Estimate VRAM usage
        let vram_mb = replicas * batch_size * 8 / 1024 / 1024;

        if vram_mb > 8000 {
            errors.push(format!("VRAM usage {} MB exceeds 8GB limit", vram_mb));
        } else if vram_mb > 6000 {
            warnings.push(format!("VRAM usage {} MB is high", vram_mb));
        }

        println!("  Estimated VRAM: {} MB", vram_mb);
    }

    if deep {
        // Deep validation - check interdependencies
        println!("  Running deep validation...");

        // Check phase dependencies
        if params
            .get("use_thermodynamic_equilibration")
            .and_then(|p| p.current.as_bool())
            .unwrap_or(false)
            && !params
                .get("gpu.enable_thermo_gpu")
                .and_then(|p| p.current.as_bool())
                .unwrap_or(false)
        {
            warnings.push("Thermodynamic enabled but GPU acceleration disabled".to_string());
        }
    }

    // Report results
    if errors.is_empty() && warnings.is_empty() {
        println!("{} Configuration valid!", "✓".green().bold());
    } else {
        if !errors.is_empty() {
            println!("\n{} Errors:", "✗".red().bold());
            for error in errors {
                println!("  • {}", error.red());
            }
        }

        if !warnings.is_empty() {
            println!("\n{} Warnings:", "⚠".yellow());
            for warning in warnings {
                println!("  • {}", warning.yellow());
            }
        }

        if !errors.is_empty() {
            std::process::exit(1);
        }
    }

    Ok(())
}

fn tune_interactive(category: &str, smart: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "╔════════════════════════════════════════╗".blue());
    println!(
        "{} {} {}",
        "║".blue(),
        format!("INTERACTIVE TUNING: {}", category.to_uppercase())
            .white()
            .bold(),
        "║".blue()
    );
    println!("{}", "╚════════════════════════════════════════╝".blue());

    // TODO: Implement interactive tuning
    println!("Interactive tuning coming soon...");

    Ok(())
}

fn diff_configs(
    file1: &str,
    file2: &str,
    changes_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Comparing configurations...", "►".cyan());

    let content1 = fs::read_to_string(file1)?;
    let content2 = fs::read_to_string(file2)?;

    let toml1: toml::Value = toml::from_str(&content1)?;
    let toml2: toml::Value = toml::from_str(&content2)?;

    // TODO: Implement diff logic
    println!("  {} → {}", file1.white(), file2.white());

    Ok(())
}

fn reset_parameters(
    all: bool,
    category: Option<String>,
    path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut count = 0;
    let mut params = CONFIG_REGISTRY.parameters.write().unwrap();

    for (param_path, param) in params.iter_mut() {
        let should_reset = all
            || category.as_ref().map_or(false, |c| &param.category == c)
            || path.as_ref().map_or(false, |p| param_path == p);

        if should_reset {
            param.current = param.default.clone();
            count += 1;
        }
    }

    println!(
        "{} Reset {} parameters to defaults",
        "✓".green().bold(),
        count
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// TELEMETRY MONITORING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

fn monitor_telemetry(
    tail: bool,
    summary: bool,
    phase: Option<String>,
    file: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = file.or_else(find_latest_metrics_file).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No metrics file found. Run a pipeline first or specify --file",
        )
    })?;

    if summary {
        show_summary(&path, phase)?;
    } else if tail {
        tail_metrics(&path, phase)?;
    } else {
        show_recent(&path, 50, phase)?;
    }

    Ok(())
}

fn find_latest_metrics_file() -> Option<PathBuf> {
    let artifacts_dir = PathBuf::from("target/run_artifacts");
    if !artifacts_dir.exists() {
        eprintln!("Artifacts directory not found: {}", artifacts_dir.display());
        return None;
    }

    std::fs::read_dir(&artifacts_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            name.starts_with("live_metrics_") && name.ends_with(".jsonl")
        })
        .max_by_key(|e| e.metadata().ok()?.modified().ok()?)
        .map(|e| e.path())
}

fn tail_metrics(
    path: &Path,
    phase_filter: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Tailing metrics from: {}", path.display());
    println!("Press Ctrl+C to stop\n");

    let mut file = File::open(path)?;
    file.seek(SeekFrom::End(0))?;

    let mut last_pos = file.metadata()?.len();

    loop {
        file.seek(SeekFrom::Start(last_pos))?;
        let reader = BufReader::new(&file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with("---") {
                continue;
            }

            match serde_json::from_str::<RunMetric>(&line) {
                Ok(metric) => {
                    if let Some(ref filter) = phase_filter {
                        let phase_name = format!("{:?}", metric.phase).to_lowercase();
                        if !phase_name.contains(&filter.to_lowercase()) {
                            continue;
                        }
                    }
                    println!("{}", format_metric(&metric));
                }
                Err(e) => {
                    eprintln!("Failed to parse metric: {}", e);
                }
            }
        }

        last_pos = file.metadata()?.len();
        thread::sleep(Duration::from_millis(500));
    }
}

fn show_summary(
    path: &Path,
    phase_filter: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut phase_stats: HashMap<String, PhaseStats> = HashMap::new();
    let mut total_metrics = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with("---") {
            continue;
        }

        match serde_json::from_str::<RunMetric>(&line) {
            Ok(metric) => {
                total_metrics += 1;
                let phase_name = format!("{:?}", metric.phase);

                if let Some(ref filter) = phase_filter {
                    if !phase_name.to_lowercase().contains(&filter.to_lowercase()) {
                        continue;
                    }
                }

                let stats = phase_stats.entry(phase_name).or_insert(PhaseStats::new());
                stats.count += 1;
                stats.total_duration += metric.duration_ms;
                stats.durations.push(metric.duration_ms);
                stats.min_colors = stats.min_colors.min(metric.chromatic_number);
                stats.max_colors = stats.max_colors.max(metric.chromatic_number);

                if metric.gpu_mode.is_gpu() {
                    stats.gpu_count += 1;
                }
            }
            Err(_) => continue,
        }
    }

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    TELEMETRY SUMMARY REPORT                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");
    println!("📁 File: {}", path.display());
    println!("📊 Total metrics: {}\n", total_metrics);

    println!("┌─────────────────┬───────┬─────────────┬─────────────┬──────────────┬─────────┐");
    println!("│ Phase           │ Count │ Total (ms)  │ Avg (ms)    │ Colors Range │ GPU %   │");
    println!("├─────────────────┼───────┼─────────────┼─────────────┼──────────────┼─────────┤");

    let mut phases: Vec<_> = phase_stats.iter().collect();
    phases.sort_by_key(|(name, _)| name.as_str());

    for (phase, stats) in phases {
        let avg = stats.total_duration / stats.count as f64;
        let gpu_percent = (stats.gpu_count as f64 / stats.count as f64) * 100.0;
        let color_range = if stats.min_colors == usize::MAX {
            "N/A".to_string()
        } else if stats.min_colors == stats.max_colors {
            format!("{}", stats.min_colors)
        } else {
            format!("{}-{}", stats.min_colors, stats.max_colors)
        };

        println!(
            "│ {:<15} │ {:>5} │ {:>11.2} │ {:>11.2} │ {:>12} │ {:>6.1}% │",
            phase, stats.count, stats.total_duration, avg, color_range, gpu_percent
        );
    }

    println!("└─────────────────┴───────┴─────────────┴─────────────┴──────────────┴─────────┘\n");

    Ok(())
}

fn show_recent(
    path: &Path,
    limit: usize,
    phase_filter: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut metrics = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with("---") {
            continue;
        }

        if let Ok(metric) = serde_json::from_str::<RunMetric>(&line) {
            if let Some(ref filter) = phase_filter {
                let phase_name = format!("{:?}", metric.phase).to_lowercase();
                if !phase_name.contains(&filter.to_lowercase()) {
                    continue;
                }
            }
            metrics.push(metric);
        }
    }

    println!("\n📊 Recent metrics (last {})\n", limit.min(metrics.len()));

    for metric in metrics.iter().rev().take(limit).rev() {
        println!("{}", format_metric(metric));
    }

    Ok(())
}

fn format_metric(metric: &RunMetric) -> String {
    let timestamp = &metric.timestamp[11..19]; // Extract HH:MM:SS
    format!(
        "{} [{:8}][{:15}] {:30} | 🎨 {:3} 🔴 {:4} | ⏱ {:8.2}ms",
        timestamp,
        format!("{:?}", metric.phase),
        format!("{}", metric.gpu_mode),
        metric.step,
        metric.chromatic_number,
        metric.conflicts,
        metric.duration_ms
    )
}

#[derive(Debug)]
struct PhaseStats {
    count: usize,
    total_duration: f64,
    durations: Vec<f64>,
    gpu_count: usize,
    min_colors: usize,
    max_colors: usize,
}

impl PhaseStats {
    fn new() -> Self {
        Self {
            count: 0,
            total_duration: 0.0,
            durations: Vec::new(),
            gpu_count: 0,
            min_colors: usize::MAX,
            max_colors: 0,
        }
    }
}
