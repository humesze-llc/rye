mod camera;
mod controller;

pub use camera::{Camera, Ray};
pub use controller::{CameraController, FirstPersonController, OrbitController};

use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraView {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}
