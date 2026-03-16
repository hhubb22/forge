use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use derive_more::From;
use serde::{Deserialize, Serialize};

use crate::{CommitConfig, Effort, ModelId, ProviderId, ReasoningConfig, SuggestConfig};

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitAuth {
    pub session_id: String,
    pub auth_url: String,
    pub token: String,
}

#[derive(Default, Clone, Serialize, Deserialize, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    pub key_info: Option<LoginInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderId>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub model: HashMap<ProviderId, ModelId>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub reasoning: HashMap<ProviderId, ReasoningPreference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<CommitConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggest: Option<SuggestConfig>,
}

impl AppConfig {
    /// Returns the configured reasoning preference for a provider.
    pub fn get_provider_reasoning(&self, provider_id: &ProviderId) -> Option<ReasoningPreference> {
        self.reasoning.get(provider_id).cloned()
    }

    /// Stores a provider-scoped reasoning preference.
    ///
    /// `inherit` is normalized by removing the explicit override, keeping the
    /// persisted configuration compact.
    pub fn set_provider_reasoning(
        &mut self,
        provider_id: ProviderId,
        reasoning: ReasoningPreference,
    ) {
        if reasoning == ReasoningPreference::Inherit {
            self.reasoning.remove(&provider_id);
        } else {
            self.reasoning.insert(provider_id, reasoning);
        }
    }

    /// Clears any provider-scoped reasoning override.
    pub fn clear_provider_reasoning(&mut self, provider_id: &ProviderId) {
        self.reasoning.remove(provider_id);
    }
}

#[derive(Clone, Serialize, Deserialize, From, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LoginInfo {
    pub api_key: String,
    pub api_key_name: String,
    pub api_key_masked: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_provider_id: Option<String>,
}

/// User-facing reasoning preference stored in application configuration.
///
/// This keeps interactive UX state separate from provider-specific request
/// payloads. Runtime code converts the preference into a `ReasoningConfig`
/// only when building a request context.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReasoningPreference {
    /// Do not persist an override; fall back to agent or workflow defaults.
    #[default]
    Inherit,
    /// Explicitly disable reasoning.
    Off,
    /// Use low reasoning effort.
    Low,
    /// Use medium reasoning effort.
    Medium,
    /// Use high reasoning effort.
    High,
    /// Use a provider-specific custom reasoning effort string.
    Custom(String),
}

impl ReasoningPreference {
    /// Returns the persisted string form of the preference.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Inherit => "inherit",
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Custom(value) => value,
        }
    }

    /// Converts the preference into a runtime reasoning configuration.
    ///
    /// `inherit` returns `None`, indicating that no user override should be
    /// applied.
    pub fn to_reasoning_config(&self) -> Option<ReasoningConfig> {
        match self {
            Self::Inherit => None,
            Self::Off => Some(ReasoningConfig::default().enabled(false)),
            Self::Low => Some(ReasoningConfig::default().enabled(true).effort(Effort::Low)),
            Self::Medium => Some(
                ReasoningConfig::default()
                    .enabled(true)
                    .effort(Effort::Medium),
            ),
            Self::High => Some(
                ReasoningConfig::default()
                    .enabled(true)
                    .effort(Effort::High),
            ),
            Self::Custom(value) => Some(
                ReasoningConfig::default()
                    .enabled(true)
                    .effort(Effort::Custom(value.clone())),
            ),
        }
    }

    /// Returns the next standard interactive preference in cycle order.
    pub fn next(self) -> Self {
        match self {
            Self::Inherit => Self::Off,
            Self::Off => Self::Low,
            Self::Low => Self::Medium,
            Self::Medium => Self::High,
            Self::High => Self::Inherit,
            Self::Custom(_) => Self::Inherit,
        }
    }
}

impl fmt::Display for ReasoningPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Serialize for ReasoningPreference {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ReasoningPreference {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::from_str(&value).map_err(serde::de::Error::custom)
    }
}

impl FromStr for ReasoningPreference {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err("Reasoning preference cannot be empty".to_string());
        }

        let normalized = trimmed.to_ascii_lowercase();
        Ok(match normalized.as_str() {
            "inherit" => Self::Inherit,
            "off" => Self::Off,
            "low" => Self::Low,
            "medium" => Self::Medium,
            "high" => Self::High,
            _ => Self::Custom(trimmed.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_reasoning_preference_custom_roundtrip() {
        let fixture = "xhigh";

        let actual = ReasoningPreference::from_str(fixture).unwrap();
        let expected = ReasoningPreference::Custom("xhigh".to_string());

        assert_eq!(actual, expected);
        assert_eq!(actual.to_string(), "xhigh");
    }

    #[test]
    fn test_reasoning_preference_to_reasoning_config_for_off() {
        let fixture = ReasoningPreference::Off;

        let actual = fixture.to_reasoning_config();
        let expected = Some(ReasoningConfig::default().enabled(false));

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_reasoning_preference_cycle_wraps() {
        let fixture = ReasoningPreference::High;

        let actual = fixture.next();
        let expected = ReasoningPreference::Inherit;

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_app_config_set_provider_reasoning_removes_inherit() {
        let mut fixture = AppConfig::default();
        let provider_id = ProviderId::OPENAI;

        fixture.set_provider_reasoning(provider_id.clone(), ReasoningPreference::High);
        fixture.set_provider_reasoning(provider_id.clone(), ReasoningPreference::Inherit);

        let actual = fixture.get_provider_reasoning(&provider_id);
        let expected = None;

        assert_eq!(actual, expected);
    }
}
