//! Admission du constructeur BVH complet (VIctory #2118).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once, OnceLock};

use super::{Shader, make};

struct Builder {
    members: HashMap<blake3::Hash, Arc<make::Pending>>,
    start: Once,
    decision: Mutex<Option<Decision>>,
}

#[derive(Clone, Copy)]
enum Decision {
    Preparing(u64),
    Final(bool),
}

fn decide(last: &mut Option<Decision>, frame: u64, status: impl FnOnce() -> Option<bool>) -> bool {
    match *last {
        Some(Decision::Final(ready)) => return ready,
        Some(Decision::Preparing(f)) if f == frame => return false,
        _ => {}
    }
    let ready = status();
    if ready.is_none() && last.is_none() {
        tracing::info!(frame, "BVH builder held: preparing pipelines");
    }
    *last = Some(ready.map_or(Decision::Preparing(frame), Decision::Final));
    ready == Some(true)
}

static BUILDER: OnceLock<Builder> = OnceLock::new();

/// Enregistre les conteneurs de la bibliothèque avant l'appel à DllMain.
pub fn register_builder<'a>(containers: impl Iterator<Item = &'a [u8]>) -> anyhow::Result<()> {
    let mut members = HashMap::new();
    for bytes in containers {
        members.entry(blake3::hash(bytes)).or_insert_with(|| {
            let shader = super::shader("compute", bytes);
            Arc::new(make::Pending::new(vec![shader], None, None))
        });
    }
    anyhow::ensure!(!members.is_empty(), "empty BVH builder inventory");
    let count = members.len();
    if let Err(other) = BUILDER.set(Builder {
        members,
        start: Once::new(),
        decision: Mutex::new(None),
    }) {
        let first = BUILDER.get().unwrap();
        anyhow::ensure!(
            first.members.len() == count
                && other
                    .members
                    .keys()
                    .all(|key| first.members.contains_key(key)),
            "BVH builder inventory changed during the session"
        );
    } else {
        tracing::info!(members = count, "BVH builder inventory");
    }
    Ok(())
}

pub(super) fn member(shaders: &[Arc<Shader>]) -> Option<Arc<make::Pending>> {
    let [cs] = shaders else { return None };
    if cs.stage != "compute" {
        return None;
    }
    let group = BUILDER.get()?;
    let host = group.members.get(&blake3::hash(&cs.container))?;
    // Une copie du conteneur et une création via un flux de sous-objets
    // retrouvent le même pipeline hôte, y compris après un Release du titre.
    group.start.call_once(|| {
        for member in group.members.values() {
            make::start_builder(member);
        }
    });
    Some(host.clone())
}

pub(super) fn admitted() -> bool {
    let group = BUILDER.get().expect("registered builder member");
    let mut last = group.decision.lock().unwrap_or_else(|e| e.into_inner());
    decide(&mut last, crate::frame::current(), || {
        #[cfg(feature = "diagnostics")]
        if group.members.keys().any(|h| {
            super::skip_hashes()
                .iter()
                .any(|s| &h.to_hex()[..16] == s.as_str())
        }) {
            tracing::info!(
                members = group.members.len(),
                "BVH builder held: member explicitly skipped"
            );
            return Some(false);
        }
        make::admit_builder(group.members.values())
    })
}

#[test]
fn builder_finishes_preparation_between_frames() {
    let mut decision = None;
    assert!(!decide(&mut decision, 3, || None));
    assert!(!decide(&mut decision, 3, || panic!("même image")));
    assert!(decide(&mut decision, 4, || Some(true)));
    assert!(decide(&mut decision, 5, || panic!("verdict définitif")));
    let mut refused = None;
    assert!(!decide(&mut refused, 3, || Some(false)));
    assert!(!decide(&mut refused, 4, || panic!("refus définitif")));
}
