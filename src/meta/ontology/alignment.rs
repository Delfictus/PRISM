use super::ConceptAnchor;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptAlignment {
    pub id: String,
    pub score: f64,
    pub attribute_overlap: f64,
    pub relation_overlap: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub target: String,
    pub primary_match: Option<ConceptAlignment>,
    pub candidates: Vec<ConceptAlignment>,
    pub explainability: String,
}

#[derive(Debug, Clone)]
pub struct AlignmentEngine {
    attribute_weight: f64,
    relation_weight: f64,
    min_score: f64,
    max_candidates: usize,
}

impl AlignmentEngine {
    pub fn new() -> Self {
        Self {
            attribute_weight: 0.6,
            relation_weight: 0.4,
            min_score: 0.35,
            max_candidates: 8,
        }
    }

    pub fn with_weights(mut self, attribute_weight: f64, relation_weight: f64) -> Self {
        let total = attribute_weight + relation_weight;
        if total > f64::EPSILON {
            self.attribute_weight = attribute_weight / total;
            self.relation_weight = relation_weight / total;
        }
        self
    }

    pub fn with_threshold(mut self, min_score: f64) -> Self {
        self.min_score = min_score.max(0.0).min(1.0);
        self
    }

    pub fn with_max_candidates(mut self, max: usize) -> Self {
        self.max_candidates = max.max(1);
        self
    }

    pub fn align(
        &self,
        target: &ConceptAnchor,
        corpus: &[ConceptAnchor],
    ) -> Result<AlignmentResult> {
        if corpus.is_empty() {
            return Err(anyhow!("ontology corpus is empty"));
        }

        let mut scored = Vec::with_capacity(corpus.len());
        let target_attrs = keys_set(target);
        let target_rels = related_set(target);

        for concept in corpus {
            if concept.id == target.id {
                continue;
            }
            let attr_overlap = jaccard(&target_attrs, &keys_set(concept));
            let rel_overlap = jaccard(&target_rels, &related_set(concept));
            let score = self.attribute_weight * attr_overlap + self.relation_weight * rel_overlap;

            if score >= self.min_score {
                scored.push(ConceptAlignment {
                    id: concept.id.clone(),
                    score,
                    attribute_overlap: attr_overlap,
                    relation_overlap: rel_overlap,
                });
            }
        }

        scored.sort_by(|a, b| match b.score.partial_cmp(&a.score) {
            Some(Ordering::Equal) | None => a.id.cmp(&b.id),
            Some(order) => order,
        });
        scored.truncate(self.max_candidates);

        let primary_match = scored.first().cloned();
        let explainability = if let Some(primary) = &primary_match {
            format!(
                "Primary alignment {} (score {:.3}) attr={:.3} rel={:.3}",
                primary.id, primary.score, primary.attribute_overlap, primary.relation_overlap
            )
        } else {
            "No candidate exceeded threshold".to_string()
        };

        Ok(AlignmentResult {
            target: target.id.clone(),
            primary_match,
            candidates: scored,
            explainability,
        })
    }
}

fn keys_set(anchor: &ConceptAnchor) -> BTreeSet<String> {
    anchor.attributes.keys().map(|k| k.to_lowercase()).collect()
}

fn related_set(anchor: &ConceptAnchor) -> BTreeSet<String> {
    anchor
        .related
        .iter()
        .map(|rel| rel.to_lowercase())
        .collect()
}

fn jaccard(lhs: &BTreeSet<String>, rhs: &BTreeSet<String>) -> f64 {
    if lhs.is_empty() && rhs.is_empty() {
        return 1.0;
    }
    let intersection = lhs.intersection(rhs).count() as f64;
    let union = (lhs.len() + rhs.len()) as f64 - intersection;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}
