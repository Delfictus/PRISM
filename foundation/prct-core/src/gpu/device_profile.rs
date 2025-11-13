//! Device Profile Configuration
//!
//! Loads and validates device profiles from TOML config.
//! Supports environment variable and CLI overrides.
//!
//! Constitutional Compliance:
//! - No stubs, full implementation
//! - Validation with explicit error messages
//! - Profile selection via PRISM_DEVICE_PROFILE env or CLI

use crate::errors::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Device profile mode
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileMode {
    /// Local single-node execution
    Local,

    /// Cluster multi-node execution
    Cluster,
}

/// Device profile strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileStrategy {
    /// Single device, single replica
    SingleDevice,

    /// Single device, multiple replicas
    SingleDeviceMultiReplica,

    /// Distributed replicas across devices
    DistributedReplicas,
}

/// Device profile specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    /// Profile mode (local or cluster)
    pub mode: ProfileMode,

    /// Number of replicas ("auto" or specific count)
    #[serde(deserialize_with = "deserialize_replicas")]
    pub replicas: ReplicaCount,

    /// Device filter patterns (e.g., ["cuda:0"], ["cuda:*"])
    pub device_filter: Vec<String>,

    /// Sync interval in milliseconds (0 = no sync)
    pub sync_interval_ms: u64,

    /// Allow remote coordination (future: gRPC/WebSocket)
    pub allow_remote: bool,

    /// Enable peer-to-peer memory access
    pub enable_peer_access: bool,

    /// Distribution strategy
    pub strategy: ProfileStrategy,

    /// Verbose logging (optional)
    #[serde(default)]
    pub verbose: bool,
}

/// Replica count specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum ReplicaCount {
    Auto,
    Fixed(usize),
}

impl ReplicaCount {
    pub fn resolve(&self, num_devices: usize) -> usize {
        match self {
            ReplicaCount::Auto => num_devices,
            ReplicaCount::Fixed(n) => *n,
        }
    }
}

/// Custom deserializer for replica count (handles "auto" or number)
fn deserialize_replicas<'de, D>(deserializer: D) -> std::result::Result<ReplicaCount, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct ReplicaCountVisitor;

    impl<'de> de::Visitor<'de> for ReplicaCountVisitor {
        type Value = ReplicaCount;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a positive integer or the string \"auto\"")
        }

        fn visit_u64<E>(self, value: u64) -> std::result::Result<ReplicaCount, E>
        where
            E: de::Error,
        {
            if value == 0 {
                Err(E::custom("replica count must be > 0"))
            } else {
                Ok(ReplicaCount::Fixed(value as usize))
            }
        }

        fn visit_i64<E>(self, value: i64) -> std::result::Result<ReplicaCount, E>
        where
            E: de::Error,
        {
            if value <= 0 {
                Err(E::custom("replica count must be > 0"))
            } else {
                Ok(ReplicaCount::Fixed(value as usize))
            }
        }

        fn visit_str<E>(self, value: &str) -> std::result::Result<ReplicaCount, E>
        where
            E: de::Error,
        {
            if value == "auto" {
                Ok(ReplicaCount::Auto)
            } else {
                Err(E::custom(format!("expected \"auto\" or a number, got \"{}\"", value)))
            }
        }
    }

    deserializer.deserialize_any(ReplicaCountVisitor)
}

impl DeviceProfile {
    /// Validate profile configuration
    pub fn validate(&self) -> Result<()> {
        if self.device_filter.is_empty() {
            return Err(PRCTError::ConfigError(
                "device_filter cannot be empty".to_string()
            ));
        }

        if let ReplicaCount::Fixed(n) = self.replicas {
            if n == 0 {
                return Err(PRCTError::ConfigError(
                    "replica count must be > 0".to_string()
                ));
            }

            if n > 256 {
                return Err(PRCTError::ConfigError(
                    format!("replica count {} exceeds maximum (256)", n)
                ));
            }
        }

        if self.sync_interval_ms > 10000 {
            return Err(PRCTError::ConfigError(
                format!("sync_interval_ms {} exceeds maximum (10000)", self.sync_interval_ms)
            ));
        }

        Ok(())
    }
}

/// Container for all device profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfiles {
    pub device_profiles: HashMap<String, DeviceProfile>,
}

impl DeviceProfiles {
    /// Load device profiles from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        let contents = std::fs::read_to_string(path)
            .map_err(|e| PRCTError::ConfigError(
                format!("Failed to read device profiles from {:?}: {}", path, e)
            ))?;

        let profiles: DeviceProfiles = toml::from_str(&contents)
            .map_err(|e| PRCTError::ConfigError(
                format!("Failed to parse device profiles TOML: {}", e)
            ))?;

        // Validate all profiles
        for (name, profile) in &profiles.device_profiles {
            profile.validate()
                .map_err(|e| PRCTError::ConfigError(
                    format!("Profile '{}' validation failed: {}", name, e)
                ))?;
        }

        Ok(profiles)
    }

    /// Get profile by name
    pub fn get(&self, name: &str) -> Option<&DeviceProfile> {
        self.device_profiles.get(name)
    }

    /// List all profile names
    pub fn list_names(&self) -> Vec<&str> {
        self.device_profiles.keys().map(|s| s.as_str()).collect()
    }
}

/// Load device profile from environment or CLI
///
/// Priority:
/// 1. CLI argument (--device-profile)
/// 2. Environment variable (PRISM_DEVICE_PROFILE)
/// 3. Default (rtx5070)
pub fn load_device_profile(
    config_path: Option<&Path>,
    cli_override: Option<&str>,
) -> Result<(String, DeviceProfile)> {
    // Determine profile path
    let profiles_path = config_path
        .unwrap_or_else(|| Path::new("foundation/prct-core/configs/device_profiles.toml"));

    println!("[DEVICE-PROFILE][LOAD] Loading profiles from {:?}", profiles_path);

    let profiles = DeviceProfiles::from_file(profiles_path)?;

    // Determine profile name (priority: CLI > ENV > default)
    let env_profile = std::env::var("PRISM_DEVICE_PROFILE").ok();
    let profile_name = cli_override
        .or_else(|| env_profile.as_deref())
        .unwrap_or("rtx5070")
        .to_string();

    println!("[DEVICE-PROFILE][LOAD] Selected profile: {}", profile_name);

    let profile = profiles.get(&profile_name)
        .ok_or_else(|| PRCTError::ConfigError(
            format!("Device profile '{}' not found. Available profiles: {:?}",
                profile_name, profiles.list_names())
        ))?
        .clone();

    profile.validate()?;

    println!("[DEVICE-PROFILE][LOAD] Profile loaded and validated: mode={:?}, replicas={:?}, strategy={:?}",
        profile.mode, profile.replicas, profile.strategy);

    Ok((profile_name, profile))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_count_deserialization() {
        let toml = r#"
            replicas = "auto"
        "#;

        #[derive(Deserialize)]
        struct TestConfig {
            #[serde(deserialize_with = "deserialize_replicas")]
            replicas: ReplicaCount,
        }

        let config: TestConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.replicas, ReplicaCount::Auto);

        let toml = r#"
            replicas = 8
        "#;

        let config: TestConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.replicas, ReplicaCount::Fixed(8));
    }

    #[test]
    fn test_profile_validation() {
        let profile = DeviceProfile {
            mode: ProfileMode::Local,
            replicas: ReplicaCount::Fixed(1),
            device_filter: vec!["cuda:0".to_string()],
            sync_interval_ms: 0,
            allow_remote: false,
            enable_peer_access: false,
            strategy: ProfileStrategy::SingleDevice,
            verbose: false,
        };

        assert!(profile.validate().is_ok());

        let invalid_profile = DeviceProfile {
            mode: ProfileMode::Local,
            replicas: ReplicaCount::Fixed(1),
            device_filter: vec![], // Empty filter
            sync_interval_ms: 0,
            allow_remote: false,
            enable_peer_access: false,
            strategy: ProfileStrategy::SingleDevice,
            verbose: false,
        };

        assert!(invalid_profile.validate().is_err());
    }
}
