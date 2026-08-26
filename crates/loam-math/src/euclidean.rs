//! Euclidean R³, the sanity-check implementation of [`Space`].

use std::borrow::Cow;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::space::{IsometryGroup, Space, WgslSpace};

/// A rigid motion of R³: a rotation followed by a translation.
///
/// Pure isometry; scale and shear are excluded by construction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso3 {
    /// Rotation about the origin, applied first. Assumed unit; `iso_apply`
    /// does not renormalize.
    pub rotation: Quat,
    /// Offset added after the rotation, in the target frame's coordinates.
    pub translation: Vec3,
}

impl Iso3 {
    /// Fixes every point; the neutral element of `iso_compose`.
    pub const IDENTITY: Self = Self {
        rotation: Quat::IDENTITY,
        translation: Vec3::ZERO,
    };

    /// Rotation about the origin, no translation.
    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            translation: Vec3::ZERO,
        }
    }

    /// Pure translation.
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            rotation: Quat::IDENTITY,
            translation,
        }
    }
}

/// Euclidean R³ with the standard metric.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanR3;

impl Space for EuclideanR3 {
    type Point = Vec3;
    type Vector = Vec3;

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        at + v
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec3, _to: Vec3, v: Vec3) -> Vec3 {
        v
    }

    fn is_chart_flat(&self) -> bool {
        true
    }
}

impl IsometryGroup for EuclideanR3 {
    type Iso = Iso3;

    fn iso_identity(&self) -> Iso3 {
        Iso3::IDENTITY
    }

    fn iso_compose(&self, a: Iso3, b: Iso3) -> Iso3 {
        // (R_a, t_a) ∘ (R_b, t_b) applied to p:
        //   R_a (R_b p + t_b) + t_a = (R_a R_b) p + (R_a t_b + t_a)
        Iso3 {
            rotation: a.rotation * b.rotation,
            translation: a.rotation * b.translation + a.translation,
        }
    }

    fn iso_inverse(&self, a: Iso3) -> Iso3 {
        let inv_rot = a.rotation.inverse();
        Iso3 {
            rotation: inv_rot,
            translation: inv_rot * (-a.translation),
        }
    }

    fn iso_apply(&self, iso: Iso3, p: Vec3) -> Vec3 {
        iso.rotation * p + iso.translation
    }

    fn iso_transport(&self, iso: Iso3, _at: Vec3, v: Vec3) -> Vec3 {
        iso.rotation * v
    }
}

impl WgslSpace for EuclideanR3 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Borrowed(WGSL_IMPL)
    }
}

// `from` is a WGSL reserved keyword (WGSL spec §3.2); use `p_from`/`p_to`.
const WGSL_IMPL: &str = r#"
// loam-math :: EuclideanR3 (v0 Space WGSL ABI)
const LOAM_MAX_ARC: f32 = 1e9;
fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32 { return length(a - b); }
fn loam_origin_distance(p: vec3<f32>) -> f32 { return length(p); }
fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> { return at + v; }
fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> { return p_to - p_from; }
fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> { return v; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn r3() -> EuclideanR3 {
        EuclideanR3
    }

    #[test]
    fn parallel_transport_along_default_impl_is_identity_in_flat_space() {
        let s = r3();
        let v = Vec3::new(1.7, -0.3, 0.5);
        assert_eq!(s.parallel_transport_along(&[], v), v);
        assert_eq!(s.parallel_transport_along(&[Vec3::ZERO], v), v);
        let path = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(2.0, 3.0, -1.0),
        ];
        let transported = s.parallel_transport_along(&path, v);
        assert_relative_eq!(transported.x, v.x);
        assert_relative_eq!(transported.y, v.y);
        assert_relative_eq!(transported.z, v.z);
    }

    #[test]
    fn iso_transport_ignores_translation() {
        let s = r3();
        let iso = Iso3::from_translation(Vec3::new(100.0, -50.0, 7.0));
        let v = Vec3::new(1.0, 0.0, 0.0);
        let transported = s.iso_transport(iso, Vec3::ZERO, v);
        assert_relative_eq!(transported.x, v.x);
        assert_relative_eq!(transported.y, v.y);
        assert_relative_eq!(transported.z, v.z);
    }
}
