//! Publishability gate over every workspace manifest.
//!
//! `cargo publish` rejects a dependency that names a `path` but no `version`,
//! and the rejection surfaces only at publish time: a path-only dep resolves
//! fine locally, so nothing in the ordinary build, clippy or test gates
//! notices that the workspace cannot be released.
//!
//! The checks are stricter than cargo's rule in two places, both deliberate.
//! Cargo strips a version-less dev-dependency from the published manifest
//! rather than rejecting it, and cargo accepts any requirement that admits the
//! local package. Requiring an exact match against `workspace.package.version`
//! keeps a version bump from leaving the hand-written requirements pointing at
//! the previous release: a dep that declares both path and version resolves
//! through the path locally, so it never disagrees with itself until publish.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item};

/// The root package's manifest directory is the workspace root.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn parse_manifest(path: &Path) -> DocumentMut {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.parse::<DocumentMut>()
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Root manifest first, then members in declaration order, so a failure
/// message names the same manifest on every run.
fn manifests() -> Vec<(String, DocumentMut)> {
    let root = parse_manifest(&workspace_root().join("Cargo.toml"));

    let members = root["workspace"]["members"]
        .as_array()
        .expect("workspace.members")
        .iter()
        .map(|member| member.as_str().expect("member path").to_owned())
        .collect::<Vec<_>>();

    let mut out = vec![("Cargo.toml".to_owned(), root)];
    for member in members {
        let path = workspace_root().join(&member).join("Cargo.toml");
        out.push((format!("{member}/Cargo.toml"), parse_manifest(&path)));
    }
    out
}

fn workspace_version() -> String {
    parse_manifest(&workspace_root().join("Cargo.toml"))["workspace"]["package"]["version"]
        .as_str()
        .expect("workspace.package.version")
        .to_owned()
}

fn member_package_names() -> BTreeSet<String> {
    manifests()
        .iter()
        .filter_map(|(_, manifest)| Some(manifest.get("package")?["name"].as_str()?.to_owned()))
        .collect()
}

/// Every dependency declared anywhere in `manifest`, as (table label, dep name, spec).
/// The three dependency kinds appear at the top level, again under each
/// `[target.'cfg(..)']` block, and once more as `[workspace.dependencies]`;
/// missing any of those would leave a place a path-only dep could hide.
fn dependencies(manifest: &DocumentMut) -> Vec<(String, String, &Item)> {
    const KINDS: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];

    let mut sections: Vec<(String, &Item)> = KINDS
        .iter()
        .filter_map(|kind| Some(((*kind).to_owned(), manifest.get(kind)?)))
        .collect();

    if let Some(section) = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("dependencies"))
    {
        sections.push(("workspace.dependencies".to_owned(), section));
    }

    if let Some(targets) = manifest.get("target").and_then(Item::as_table_like) {
        for (cfg, table) in targets.iter() {
            for kind in KINDS {
                if let Some(section) = table.get(kind) {
                    sections.push((format!("target.{cfg}.{kind}"), section));
                }
            }
        }
    }

    let mut out = Vec::new();
    for (label, section) in sections {
        for (name, spec) in section.as_table_like().expect("dependency table").iter() {
            out.push((label.clone(), name.to_owned(), spec));
        }
    }
    out
}

fn dep_key<'a>(spec: &'a Item, key: &str) -> Option<&'a Item> {
    spec.as_table_like()?.get(key)
}

/// A path dep with no version is exactly what `cargo publish` refuses.
#[test]
fn path_dependencies_declare_a_version() {
    let mut offenders = Vec::new();
    for (manifest_label, manifest) in manifests() {
        for (table, name, spec) in dependencies(&manifest) {
            if dep_key(spec, "path").is_some() && dep_key(spec, "version").is_none() {
                offenders.push(format!("{manifest_label} [{table}] {name}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "path dependencies without a version make `cargo publish` fail:\n  {}",
        offenders.join("\n  "),
    );
}

/// Pins the version requirements against `workspace.package.version` so a
/// version bump cannot silently leave them behind.
#[test]
fn intra_workspace_dep_versions_track_the_workspace_version() {
    let expected = workspace_version();
    let members = member_package_names();

    let mut offenders = Vec::new();
    for (manifest_label, manifest) in manifests() {
        for (table, name, spec) in dependencies(&manifest) {
            if !members.contains(&name) {
                continue;
            }
            let version = dep_key(spec, "version").and_then(Item::as_str);
            if version != Some(expected.as_str()) {
                offenders.push(format!(
                    "{manifest_label} [{table}] {name}: {version:?}, expected {expected:?}"
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "intra-workspace dep versions disagree with workspace.package.version:\n  {}",
        offenders.join("\n  "),
    );
}
