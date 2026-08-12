//! `sdf` scene: author a signed-distance scene while it is being rendered.
//!
//! The scene owns a live [`loam_scene::Scene`] and re-emits, recompiles and
//! rebuilds its render node on the frames where an edit landed. Every constant
//! in the tree is a baked WGSL literal ([`loam_scene::Primitive::to_wgsl`]), so
//! an edit is a shader compile, and `examples/sdf_edit_latency.rs` is what that
//! costs. Measured on an idle RTX 4090 Laptop over 120 single-parameter edits:
//! emit p50 0.009ms, shader module 0.352ms, pipeline 0.319ms, total per edit
//! p50 0.685ms and p95 0.850ms, which is 5% of a 16.7ms frame. Emitting is
//! free at the scale of a frame and the pipeline build costs about 35x it, not
//! the three orders of magnitude an earlier run under compile contention
//! reported. A drag therefore has headroom rather than sitting on the frame
//! budget. Binding the selected leaf's constants to `RayMarchUniforms::params`
//! would make a drag write a uniform instead of a module, but on these numbers
//! it buys margin rather than rescuing the design, so it is a node of its own
//! and not an urgent one.
//!
//! The module assembles WGSL by hand, the way [`crate::Demo::new`] does:
//! `wgsl_impl` + scene emit + march kernel + the shading fragment below, then
//! `create_shader_module`. `ShaderDb` is not in the path because it reads
//! source from disk, which is nothing in a browser.
//!
//! ## The interaction route, and why it is this one
//!
//! Nothing in the panel writes the tree. A widget's only output is a
//! [`SceneEdit`], and the only way an edit reaches the tree is
//! [`loam_app::command::submit`], the runner-owned queue that drains once per
//! frame ahead of the ticks. The panel, a console line and a `--script` line
//! are therefore the same caller, and `sdf set root.0 radius 0.2` typed into
//! the console is exactly what dragging that slider produces. A typed
//! [`SceneEdit`] and a command line are not two surfaces here:
//! [`SceneEdit::to_args`] and [`SceneEdit::from_args`] round trip bit-exactly,
//! so the value is the transport and the text is its spelling.
//!
//! Selection and the slider draft are the deliberate exceptions, and they are
//! not mutations of the document: they are written inline because a click that
//! took a frame to open a parameter panel would read as lag, and because a
//! slider whose value is re-read from the tree mid-drag snaps back on the frame
//! between the drag and the drain. The draft resyncs from the tree whenever
//! the selection moves or an edit lands, so a console `sdf set` shows up under
//! the cursor.
//!
//! Two known egui frictions are answered rather than inherited: values are
//! drawn through [`loam_egui::slider_with_edit`], whose fixed-width value cell
//! keeps a column of sliders from jittering as digits change, and the deferral
//! here is a command-queue property rather than the playground's historical
//! two-pass-overlay workaround.

use std::borrow::Cow;

use anyhow::{Context, Result};
use glam::Vec3;
use loam_app::command::{CommandCtx, CommandLine};
use loam_app::{egui, Camera, CameraController, FrameCtx, OrbitController, RenderCtx, SetupCtx};
use loam_egui::{cmd, Console, ConsoleUi};
use loam_math::{EuclideanR3, WgslSpace};
use loam_render::{GeodesicRayMarchNode, Viewport};
use loam_scene::edit::{
    self, Combinator, EditValue, LeafKind, NodePath, Param, SceneEdit, DEFAULT_BLEND_RADIUS,
};
use loam_scene::{Scene, SceneNode};
use loam_shader::GEODESIC_MARCH_KERNEL;

/// Panel width in egui points. Wide enough for the deepest boot-scene row plus
/// its label without wrapping, which is what keeps the tree readable as a
/// tree.
const PANEL_WIDTH: f32 = 268.0;

/// Fixed width of a slider's value cell, in points. Same purpose as the
/// control panel's: the slider's right edge must not move as digits change.
const VALUE_CELL_WIDTH: f32 = 62.0;

const BOOT_ORBIT_DISTANCE: f32 = 4.2;
const BOOT_ORBIT_PITCH: f32 = -0.22;

/// Shading half of the four-layer assembly. Deliberately plain: this scene is
/// about the field the user is authoring, so the only cues are a key light, a
/// hemisphere fill that keeps away-facing surfaces readable, and depth fog.
const SHADING_WGSL: &str = r#"
struct RayMarchUniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    // params.xy is the viewport origin in pixels; the side panel offsets the
    // march region but @builtin(position) stays framebuffer-relative.
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: RayMarchUniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

const HORIZON: vec3<f32> = vec3<f32>(0.055, 0.060, 0.075);
const ZENITH: vec3<f32> = vec3<f32>(0.012, 0.014, 0.022);
const SURFACE: vec3<f32> = vec3<f32>(0.62, 0.65, 0.72);

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (frag_pos.xy - u.params.xy) / u.resolution;
    let ndc = uv * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let sky = mix(ZENITH, HORIZON, uv.y);
    let dir = loam_safe_normalize(
        u.camera_forward
            + u.camera_right * ndc.x * u.fov_y_tan * aspect
            - u.camera_up * ndc.y * u.fov_y_tan,
        vec3<f32>(0.0, 0.0, -1.0),
    );

    let hit = loam_march_geodesic(u.camera_pos, dir, 1.0);
    if hit.w < 0.0 {
        return vec4<f32>(sky, 1.0);
    }

    let normal = loam_estimate_normal(hit.xyz, 1.0);
    let key = normalize(vec3<f32>(0.42, 0.83, 0.37));
    let lambert = max(dot(normal, key), 0.0) * 0.78;
    let fill = 0.18 * (0.5 + 0.5 * normal.y);
    let fog = exp(-0.055 * hit.w);
    return vec4<f32>(mix(sky, SURFACE * (lambert + fill + 0.16), fog), 1.0);
}
"#;

/// Four leaves and every combinator, so the panel has one of each to edit on
/// the first frame: `((sphere ~ box) | plane) - sphere`.
fn boot_scene() -> Scene {
    Scene::new(
        SceneNode::sphere(Vec3::new(-0.35, 0.0, 0.0), 0.45)
            .smooth_union(SceneNode::box_(Vec3::splat(0.4)), DEFAULT_BLEND_RADIUS)
            .union(SceneNode::plane(Vec3::Y, -0.8))
            .subtract(SceneNode::sphere(Vec3::new(0.3, 0.35, 0.35), 0.35)),
    )
}

/// The four-layer WGSL the node is compiled from. A free function so the
/// assembly is checkable without a device.
fn assemble(scene: &Scene) -> String {
    format!(
        "{prelude}\n{emit}\n{kernel}\n{shading}",
        prelude = EuclideanR3.wgsl_impl(),
        emit = scene.to_wgsl(&EuclideanR3),
        kernel = GEODESIC_MARCH_KERNEL,
        shading = SHADING_WGSL,
    )
}

/// The document: the tree, plus a counter that identifies its version.
///
/// This is the console's `Ctx`, so a console line and a panel widget reach the
/// same state through the same registry entry. Everything that mutates it goes
/// through [`edit::apply`].
pub(crate) struct Editor {
    scene: Scene,
    /// Bumped once per landed edit. The render node is rebuilt when it falls
    /// behind, which is what makes a recompile a consequence of a change
    /// rather than of a frame.
    generation: u64,
}

impl Editor {
    fn apply(&mut self, edit: &SceneEdit) -> Result<bool> {
        let changed = edit::apply(&mut self.scene, edit)?;
        if changed {
            self.generation += 1;
        }
        Ok(changed)
    }

    fn replace(&mut self, scene: Scene) {
        self.scene = scene;
        self.generation += 1;
    }
}

/// `sdf`: the whole mutation and serialisation surface, as one console verb.
/// The panel submits the same `set` / `add` / `remove` lines.
fn register_commands(console: &mut Console<Editor>) {
    console.register(
        cmd::<Editor, _>(
            "sdf",
            "edit the live SDF scene; bare `sdf` lists the tree by path",
            |args, editor: &mut Editor, out| {
                match args.first().copied() {
                    None => {
                        edit::for_each_node(&editor.scene.root, |path, node| {
                            let indent = "  ".repeat(path.depth());
                            out.line(format!("{indent}{path}  {}", edit::label(node)));
                        });
                    }
                    Some("export") => {
                        for line in editor.scene.to_ron()?.lines() {
                            out.line(line.to_string());
                        }
                    }
                    Some("save") => {
                        let path = args.get(1).context("sdf save <path.ron>")?;
                        std::fs::write(path, editor.scene.to_ron()?)
                            .with_context(|| format!("writing {path}"))?;
                        out.line(format!("sdf: wrote {path}"));
                    }
                    Some("load") => {
                        let path = args.get(1).context("sdf load <path.ron>")?;
                        editor.replace(Scene::load(path)?);
                        out.line(format!("sdf: loaded {path}"));
                    }
                    _ => {
                        let edit = SceneEdit::from_args(args)?;
                        if editor.apply(&edit)? {
                            out.line(format!("sdf: {}", edit.to_args().join(" ")));
                        }
                    }
                }
                Ok(())
            },
        )
        .with_args(&[&["set", "add", "remove", "export", "save", "load"]])
        .with_long_help(
            "Edits the scene the raymarch is drawing. Every panel widget emits one\n\
             of these lines, so anything the panel can do is scriptable.\n\
             \n\
             sdf                          list the tree, one node per path\n\
             sdf set <path> <param> <v..> retune a parameter (center, radius,\n  \
                                          half-extents, normal, offset, blend)\n\
             sdf add <path> <op> <kind>   wrap the node at <path> in op(node, kind);\n  \
                                          op is union|intersection|difference|\n  \
                                          smooth-union, kind is sphere|box|plane\n\
             sdf remove <path>            drop a node, collapsing its parent into\n  \
                                          the surviving sibling\n\
             sdf export                   print the scene as RON\n\
             sdf save <path.ron>          write that RON to a file (native only)\n\
             sdf load <path.ron>          replace the scene from a file\n\
             \n\
             Paths are positional: `root`, `root.0`, `root.1.0`. A structural edit\n\
             moves them, so re-list after an add or a remove.",
        ),
    );
}

/// Slider bounds per parameter. A range is a UI decision, not a property of
/// the tree, so it lives here rather than in `loam-scene`; the precise-edit
/// popup on each value cell reaches past it.
fn slider_range(param: Param) -> std::ops::RangeInclusive<f32> {
    match param {
        Param::Center | Param::Offset => -2.0..=2.0,
        Param::Radius | Param::HalfExtents => 0.01..=1.5,
        Param::Normal => -1.0..=1.0,
        Param::Blend => 0.01..=0.5,
    }
}

/// The selected node's parameter values while the user is dragging them.
///
/// Held outside the tree so the widgets have somewhere to write that is not
/// the document, and resynced from the tree when the selection moves or an
/// edit lands.
#[derive(Default)]
struct Draft {
    path: NodePath,
    generation: u64,
    values: Vec<(Param, EditValue)>,
}

impl Draft {
    fn sync(&mut self, node: &SceneNode, path: &NodePath, generation: u64) {
        if self.path != *path || self.generation != generation {
            self.values = edit::parameters(node);
            self.path = path.clone();
            self.generation = generation;
        }
    }
}

/// What the add row is currently set to build.
struct AddChoice {
    combinator: Combinator,
    leaf: LeafKind,
}

impl Default for AddChoice {
    fn default() -> Self {
        Self {
            combinator: Combinator::Union,
            leaf: LeafKind::Sphere,
        }
    }
}

/// One row per node, indented by depth, the variant as the click target.
/// Writes only the selection.
fn tree_panel(ui: &mut egui::Ui, scene: &Scene, selected: &mut NodePath) {
    edit::for_each_node(&scene.root, |path, node| {
        ui.horizontal(|ui| {
            ui.add_space(12.0 * path.depth() as f32);
            // The hover text is the node's address, which is what a console
            // line needs and the only place it is discoverable.
            if ui
                .selectable_label(*selected == *path, edit::label(node))
                .on_hover_text(path.to_string())
                .clicked()
            {
                *selected = path.clone();
            }
        });
    });
}

/// Add and remove, as one row each. Pushes edits; writes nothing.
fn structure_panel(
    ui: &mut egui::Ui,
    scene: &Scene,
    selected: &NodePath,
    choice: &mut AddChoice,
    out: &mut Vec<SceneEdit>,
) {
    let resolves = edit::node_at(&scene.root, selected).is_some();
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("sdf-add-op")
            .width(96.0)
            .selected_text(choice.combinator.name())
            .show_ui(ui, |ui| {
                for op in Combinator::ALL {
                    ui.selectable_value(&mut choice.combinator, op, op.name());
                }
            });
        egui::ComboBox::from_id_salt("sdf-add-leaf")
            .width(72.0)
            .selected_text(choice.leaf.name())
            .show_ui(ui, |ui| {
                for kind in LeafKind::ALL {
                    ui.selectable_value(&mut choice.leaf, kind, kind.name());
                }
            });
        if ui
            .add_enabled(resolves, egui::Button::new("add"))
            .on_hover_text("wrap the selected node; the new leaf becomes the selection")
            .clicked()
        {
            out.push(SceneEdit::Insert {
                path: selected.clone(),
                combinator: choice.combinator,
                leaf: choice.leaf,
            });
        }
    });
    if ui
        .add_enabled(
            resolves && selected.depth() > 0,
            egui::Button::new("remove selected"),
        )
        .on_hover_text("the sibling takes its parent's place")
        .clicked()
    {
        out.push(SceneEdit::Remove {
            path: selected.clone(),
        });
    }
}

/// Sliders for the selected node's parameters. Every change becomes an edit in
/// `out`; the tree is not touched.
fn parameter_panel(
    ui: &mut egui::Ui,
    node: &SceneNode,
    path: &NodePath,
    generation: u64,
    draft: &mut Draft,
    out: &mut Vec<SceneEdit>,
) {
    draft.sync(node, path, generation);
    if draft.values.is_empty() {
        ui.label(
            egui::RichText::new(format!("{} has no editable constants", edit::label(node)))
                .small()
                .weak(),
        );
        return;
    }
    for (param, value) in &mut draft.values {
        let range = slider_range(*param);
        ui.label(egui::RichText::new(param.name()).small().weak());
        let changed = match value {
            EditValue::Scalar(v) => scalar_row(ui, "", v, range),
            EditValue::Vector(v) => {
                let mut any = false;
                for (axis, component) in ["x", "y", "z"].into_iter().zip(v.as_mut()) {
                    any |= scalar_row(ui, axis, component, range.clone());
                }
                any
            }
        };
        if changed {
            out.push(SceneEdit::Set {
                path: path.clone(),
                param: *param,
                value: *value,
            });
        }
    }
}

fn scalar_row(
    ui: &mut egui::Ui,
    axis: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        if !axis.is_empty() {
            ui.label(egui::RichText::new(axis).small().monospace());
        }
        let formatted = format!("{value:+.3}");
        changed =
            loam_egui::slider_with_edit(ui, value, range, &formatted, "", 3, VALUE_CELL_WIDTH)
                .changed;
    });
    changed
}

pub(crate) struct SdfScene {
    camera: Camera<EuclideanR3>,
    orbit: OrbitController<EuclideanR3>,
    console: Console<Editor>,
    editor: Editor,
    node: GeodesicRayMarchNode,
    /// Generation the compiled module was emitted from.
    compiled: u64,
    /// View state, one frame ahead of the tree after a structural edit: the
    /// address the edit will have created by the time the queue drains.
    selected: NodePath,
    draft: Draft,
    add: AddChoice,
    /// Edits the panel produced this frame, drained into the command queue
    /// once it returns. Held on the scene rather than in `ui` so the buffer is
    /// reused across frames.
    pending: Vec<SceneEdit>,
    /// `[x, y, width, height]` in physical pixels of whatever the menu bar and
    /// the side panel left, read back from egui's own rect rather than
    /// recomputed from the panel constant, so the march is centred in the
    /// region the user can actually see.
    region: [u32; 4],
    last_egui_keyboard: bool,
}

impl SdfScene {
    pub(crate) fn new(ctx: &mut SetupCtx<'_>) -> Result<Self> {
        let mut console = Console::<Editor>::new();
        loam_app::shell::register_command::<Editor, crate::shell::Playground>(&mut console);
        register_commands(&mut console);

        let editor = Editor {
            scene: boot_scene(),
            generation: 0,
        };
        let node = compile(ctx.rd, &editor.scene);

        let mut camera = Camera::<EuclideanR3>::at_origin();
        camera.position = Vec3::new(0.0, 1.0, BOOT_ORBIT_DISTANCE);
        let mut orbit: OrbitController<EuclideanR3> = OrbitController::default();
        orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);

        Ok(Self {
            camera,
            orbit,
            console,
            editor,
            node,
            compiled: 0,
            selected: NodePath::root(),
            draft: Draft::default(),
            add: AddChoice::default(),
            pending: Vec::new(),
            region: [0; 4],
            last_egui_keyboard: false,
        })
    }

    fn viewport(&self, rd: &loam_render::device::RenderDevice) -> Viewport {
        let cfg = &rd.surface_bundle.config;
        march_viewport(self.region, [cfg.width, cfg.height])
    }
}

/// The march region, clamped into the framebuffer.
///
/// egui's rect is a frame behind the swapchain across a resize and is empty
/// before the first UI pass; a viewport reaching outside the attachment is a
/// wgpu validation failure rather than a clipped draw, so neither case is
/// allowed to reach `set_viewport`.
fn march_viewport(region: [u32; 4], framebuffer: [u32; 2]) -> Viewport {
    let [width, height] = framebuffer.map(|d| d.max(1));
    let full = Viewport {
        x: 0,
        y: 0,
        width,
        height,
    };
    let [x, y, w, h] = region;
    if w == 0 || h == 0 {
        return full;
    }
    let x = x.min(width - 1);
    let y = y.min(height - 1);
    Viewport {
        x,
        y,
        width: w.min(width - x),
        height: h.min(height - y),
    }
}

/// Emit, compile, and build the node. Separate from the scene so the rebuild
/// path and the boot path cannot drift.
fn compile(rd: &loam_render::device::RenderDevice, scene: &Scene) -> GeodesicRayMarchNode {
    let module = rd
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sdf editor"),
            source: wgpu::ShaderSource::Wgsl(assemble(scene).into()),
        });
    GeodesicRayMarchNode::from_module(&rd.device, rd.target_format(), &module, rd.sample_count())
}

impl loam_app::shell::Scene for SdfScene {
    /// Every mutation this scene has arrives here, from the one queue the
    /// runner drains before the frame's ticks.
    fn apply_command(&mut self, cmd: &CommandLine, _ctx: &mut CommandCtx<'_>) -> Result<()> {
        self.console
            .dispatch(&cmd.name, &cmd.arg_refs(), &mut self.editor);
        Ok(())
    }

    fn update(&mut self, ctx: &mut FrameCtx<'_>) {
        if self.compiled != self.editor.generation {
            // Carried by value across the rebuild: the camera pose and the
            // viewport are frame state, not shader state, and re-deriving them
            // here would put a one-frame jump on every edit.
            let uniforms = *self.node.uniforms();
            self.node = compile(ctx.rd, &self.editor.scene);
            self.node.set_uniforms(&ctx.rd.queue, uniforms);
            self.compiled = self.editor.generation;
        }

        let viewport = self.viewport(ctx.rd);
        self.camera.aspect = viewport.width as f32 / viewport.height.max(1) as f32;
        if !ctx.ui_has_focus {
            self.orbit
                .advance(ctx.input, &mut self.camera, &EuclideanR3, ctx.dt);
        }

        let view = self.camera.view();
        let uniforms = self.node.uniforms_mut();
        uniforms.camera_pos = view.position.into();
        uniforms.camera_forward = view.forward.into();
        uniforms.camera_right = view.right.into();
        uniforms.camera_up = view.up.into();
        uniforms.fov_y_tan = (self.camera.fov_y * 0.5).tan();
        uniforms.resolution = viewport.resolution_f32();
        uniforms.time = ctx.time;
        uniforms.params = [viewport.x as f32, viewport.y as f32, 0.0, 0.0];
        self.node.flush_uniforms(&ctx.rd.queue);
    }

    fn ui(&mut self, ctx: &egui::Context, _frame: &mut FrameCtx<'_>) {
        self.pending.clear();
        egui::SidePanel::left("sdf-editor")
            .exact_width(PANEL_WIDTH)
            .resizable(false)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    tree_panel(ui, &self.editor.scene, &mut self.selected);
                    ui.separator();
                    structure_panel(
                        ui,
                        &self.editor.scene,
                        &self.selected,
                        &mut self.add,
                        &mut self.pending,
                    );
                    ui.separator();
                    match edit::node_at(&self.editor.scene.root, &self.selected) {
                        Some(node) => parameter_panel(
                            ui,
                            node,
                            &self.selected,
                            self.editor.generation,
                            &mut self.draft,
                            &mut self.pending,
                        ),
                        None => {
                            ui.label(egui::RichText::new("selection pending").small().weak());
                        }
                    }
                });
            });
        // Read after the panel is added, so it is what the menu bar and the
        // panel left rather than what they were asked for.
        let free = ctx.available_rect();
        let scale = ctx.pixels_per_point();
        self.region = [
            free.min.x * scale,
            free.min.y * scale,
            free.width() * scale,
            free.height() * scale,
        ]
        .map(|edge| edge.max(0.0).round() as u32);

        for edit in self.pending.drain(..) {
            self.selected = edit.focus_after();
            loam_app::command::submit(CommandLine {
                name: "sdf".to_string(),
                args: edit.to_args(),
            });
        }

        loam_app::log::pump_into(&mut self.console);
        loam_app::command::pump_into(&mut self.console);
        self.console.ui(ctx);
        loam_app::command::forward_pending(&mut self.console);
        self.last_egui_keyboard = ctx.wants_keyboard_input();
    }

    fn on_key(
        &mut self,
        code: winit::keyboard::KeyCode,
        state: winit::event::ElementState,
        _ctx: &mut FrameCtx<'_>,
    ) {
        use winit::event::ElementState;
        use winit::keyboard::KeyCode;
        if self.last_egui_keyboard || state != ElementState::Pressed {
            return;
        }
        if code == KeyCode::KeyR {
            self.orbit.set_orbit(BOOT_ORBIT_DISTANCE, BOOT_ORBIT_PITCH);
        }
    }

    fn record(&mut self, ctx: &mut RenderCtx<'_>) -> Result<()> {
        let viewport = self.viewport(ctx.rd);
        self.node
            .record_in_viewport(ctx.encoder, ctx.view, viewport);
        Ok(())
    }

    fn title(&self, _fps: f32) -> Cow<'static, str> {
        Cow::Borrowed("polytope playground - SDF editor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loam_scene::edit::EditValue;
    use loam_shader::validate_wgsl;

    fn path(text: &str) -> NodePath {
        text.parse().expect("path")
    }

    fn screen() -> egui::RawInput {
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_max(
                egui::pos2(0.0, 0.0),
                egui::pos2(1280.0, 720.0),
            )),
            time: Some(0.0),
            ..Default::default()
        }
    }

    /// The panel is only worth opening if the boot tree has one of everything
    /// to open it on: four leaves and all four combinators.
    #[test]
    fn the_boot_scene_carries_every_combinator_and_an_editable_leaf_of_each_kind() {
        let scene = boot_scene();
        let mut labels = Vec::new();
        edit::for_each_node(&scene.root, |_, node| labels.push(edit::label(node)));
        for expected in [
            "difference",
            "union",
            "smooth-union",
            "sphere",
            "box",
            "plane",
        ] {
            assert!(labels.contains(&expected), "boot scene lacks {expected}");
        }
        for kind in LeafKind::ALL {
            assert!(labels.contains(&kind.name()));
        }
    }

    /// The property behind "a primitive can be added, moved and removed
    /// without recompiling the binary": the shader text the editor assembles
    /// after an arbitrary run of edits is still WGSL a validator accepts. A
    /// device is not needed to know that, and this is the closest a headless
    /// test can stand to the compile the scene performs each time an edit
    /// lands.
    #[test]
    fn the_assembled_shader_stays_valid_wgsl_through_add_move_and_remove() {
        let mut scene = boot_scene();
        validate_wgsl(&assemble(&scene)).expect("the boot scene assembles");

        let script = [
            SceneEdit::Insert {
                path: path("root.0"),
                combinator: Combinator::SmoothUnion,
                leaf: LeafKind::Sphere,
            },
            SceneEdit::Set {
                path: path("root.0.1"),
                param: Param::Center,
                value: EditValue::Vector(Vec3::new(0.55, -0.2, 0.31)),
            },
            SceneEdit::Set {
                path: path("root.0.1"),
                param: Param::Radius,
                value: EditValue::Scalar(0.21),
            },
            SceneEdit::Insert {
                path: path("root.1"),
                combinator: Combinator::Union,
                leaf: LeafKind::Plane,
            },
            SceneEdit::Set {
                path: path("root.1.1"),
                param: Param::Normal,
                value: EditValue::Vector(Vec3::new(0.3, 0.9, -0.2)),
            },
            SceneEdit::Remove {
                path: path("root.1.1"),
            },
            SceneEdit::Insert {
                path: path("root"),
                combinator: Combinator::Difference,
                leaf: LeafKind::Box,
            },
        ];
        for edit in script {
            assert!(
                edit::apply(&mut scene, &edit).expect("the script applies"),
                "{:?} landed nothing",
                edit.to_args(),
            );
            validate_wgsl(&assemble(&scene))
                .unwrap_or_else(|e| panic!("after {:?}: {e}", edit.to_args()));
        }
    }

    /// Save and load is a real round trip through the file boundary: what the
    /// editor exports is what `Scene::load` reads back, down to the emitted
    /// WGSL. `to_ron` is what `sdf save` writes and `Scene::load` what `sdf
    /// load` calls, so this is the command pair and not a proxy for it.
    #[test]
    fn an_edited_scene_survives_the_save_and_load_the_commands_perform() {
        let mut scene = boot_scene();
        for edit in [
            SceneEdit::Insert {
                path: path("root.0.0"),
                combinator: Combinator::Union,
                leaf: LeafKind::Box,
            },
            SceneEdit::Set {
                path: path("root.0.0.1"),
                param: Param::HalfExtents,
                value: EditValue::Vector(Vec3::new(0.11, 0.42, 0.7)),
            },
            SceneEdit::Remove {
                path: path("root.1"),
            },
        ] {
            edit::apply(&mut scene, &edit).expect("applies");
        }

        let file = std::env::temp_dir().join("loam-sdf-editor-round-trip.ron");
        std::fs::write(&file, scene.to_ron().expect("serialize")).expect("write");
        let loaded = Scene::load(&file).expect("load");
        std::fs::remove_file(&file).expect("clean up");

        assert_eq!(
            assemble(&loaded),
            assemble(&scene),
            "a saved scene must reload as the same shader",
        );
    }

    /// An immediate-mode panel that reports a change on a frame nobody touched
    /// would submit an edit, bump the generation and recompile the shader
    /// every frame. Drawn twice, because egui's first frame has no layout to
    /// hit-test against and would hide a widget that fires on hover.
    #[test]
    fn a_frame_with_no_input_produces_no_edits() {
        let ctx = egui::Context::default();
        let scene = boot_scene();
        let mut draft = Draft::default();
        let mut add = AddChoice::default();
        let mut selected = path("root.0.0.0");
        let mut pending = Vec::new();
        for _ in 0..2 {
            let _ = ctx.run(screen(), |ctx| {
                egui::SidePanel::left("test").show(ctx, |ui| {
                    tree_panel(ui, &scene, &mut selected);
                    structure_panel(ui, &scene, &selected, &mut add, &mut pending);
                    let node = edit::node_at(&scene.root, &selected).expect("selected");
                    parameter_panel(ui, node, &selected, 0, &mut draft, &mut pending);
                });
            });
            assert!(pending.is_empty(), "phantom edits: {pending:?}");
        }
        assert_eq!(selected, path("root.0.0.0"), "selection must not drift");
    }

    /// The tree list is the only place a path is discoverable, so a walk that
    /// skipped a node would hide it from the editor entirely. Asserted as
    /// height growing by exactly one row pitch per node across three trees
    /// rather than against a hardcoded pitch, which is an egui style value.
    /// A panel that drew only leaves, or only interior nodes, is not linear in
    /// the node count and fails here.
    #[test]
    fn the_tree_panel_grows_by_exactly_one_row_per_node() {
        let ctx = egui::Context::default();
        let height = |scene: &Scene| -> f32 {
            let mut selected = NodePath::root();
            let mut grown = 0.0;
            let _ = ctx.run(screen(), |ctx| {
                egui::SidePanel::left("test").show(ctx, |ui| {
                    let before = ui.next_widget_position().y;
                    tree_panel(ui, scene, &mut selected);
                    grown = ui.next_widget_position().y - before;
                });
            });
            grown
        };
        let leaf = || SceneNode::sphere(Vec3::ZERO, 0.1);
        let cases = [
            Scene::new(leaf()),
            Scene::new(leaf().union(leaf())),
            boot_scene(),
        ];
        let mut counts = vec![0usize; cases.len()];
        for (scene, count) in cases.iter().zip(&mut counts) {
            edit::for_each_node(&scene.root, |_, _| *count += 1);
        }
        assert_eq!(counts, vec![1, 3, 7], "fixture node counts");

        let heights: Vec<f32> = cases.iter().map(height).collect();
        let pitch = (heights[1] - heights[0]) / 2.0;
        assert!(pitch > 1.0, "a row must have height, got {pitch}");
        assert!(
            (heights[2] - heights[1] - 4.0 * pitch).abs() < 0.5,
            "seven nodes must be four rows taller than three: {heights:?} at pitch {pitch}",
        );
    }

    /// The panel's rect is a frame behind the swapchain across a resize, so
    /// the march region has to be clamped rather than trusted: a viewport
    /// reaching past the attachment is a validation failure that takes the
    /// process with it. Swept over a shrink in each axis and both, plus the
    /// pre-first-frame empty rect.
    #[test]
    fn the_march_viewport_never_leaves_the_framebuffer() {
        let stale = [268, 24, 1012, 696];
        for framebuffer in [
            [1280, 720],
            [640, 720],
            [1280, 360],
            [200, 100],
            [1, 1],
            [0, 0],
        ] {
            for region in [stale, [0; 4], [0, 0, 1, 1], [4000, 4000, 32, 32]] {
                let v = march_viewport(region, framebuffer);
                let [width, height] = framebuffer.map(|d| d.max(1));
                assert!(v.width > 0 && v.height > 0, "{region:?} in {framebuffer:?}");
                assert!(
                    v.x + v.width <= width && v.y + v.height <= height,
                    "{v:?} escapes {framebuffer:?}",
                );
            }
        }
        // An unset region is the whole framebuffer, which is what the first
        // frame draws before any UI pass has run.
        assert_eq!(
            march_viewport([0; 4], [1280, 720]),
            Viewport::full([1280, 720])
        );
    }

    /// The draft is what the sliders write to, so it has to hold the node's
    /// own values, follow the selection, and follow an edit that landed from
    /// somewhere else (a console line). Between those it must stay put, which
    /// is what stops a drag from snapping back on the frame the queue drains.
    #[test]
    fn the_draft_follows_the_selection_and_landed_edits_and_nothing_else() {
        let scene = boot_scene();
        let sphere = path("root.1");
        let node = edit::node_at(&scene.root, &sphere).expect("sphere");
        let mut draft = Draft::default();
        draft.sync(node, &sphere, 0);
        assert_eq!(draft.values, edit::parameters(node));

        // A drag writes the draft; a resync at the same generation and path
        // must leave it alone.
        draft.values[1] = (Param::Radius, EditValue::Scalar(0.9));
        draft.sync(node, &sphere, 0);
        assert_eq!(draft.values[1], (Param::Radius, EditValue::Scalar(0.9)));

        // The edit landed: the tree is now authoritative again.
        draft.sync(node, &sphere, 1);
        assert_eq!(draft.values, edit::parameters(node));

        // Selecting another node reloads even at the same generation.
        let plane = path("root.0.1");
        let node = edit::node_at(&scene.root, &plane).expect("plane");
        draft.sync(node, &plane, 1);
        assert_eq!(draft.values, edit::parameters(node));
    }

    /// Every parameter the panel can draw has a slider range that contains the
    /// value a freshly inserted leaf starts at. A default outside its own
    /// range would be clamped to something else the moment the panel drew it,
    /// which is an edit the user did not make.
    #[test]
    fn every_default_leaf_value_lies_inside_its_slider_range() {
        for kind in LeafKind::ALL {
            let node = SceneNode::Leaf(kind.shape());
            for (param, value) in edit::parameters(&node) {
                let range = slider_range(param);
                for component in value.components() {
                    assert!(
                        range.contains(component),
                        "{}'s {} starts at {component} outside {range:?}",
                        kind.name(),
                        param.name(),
                    );
                }
            }
        }
        let blend = SceneNode::sphere(Vec3::ZERO, 0.1)
            .smooth_union(SceneNode::sphere(Vec3::X, 0.1), DEFAULT_BLEND_RADIUS);
        assert!(slider_range(Param::Blend).contains(&DEFAULT_BLEND_RADIUS));
        assert_eq!(
            edit::parameters(&blend),
            vec![(Param::Blend, EditValue::Scalar(DEFAULT_BLEND_RADIUS))],
        );
    }

    /// The route the module doc claims: a widget's edit and a typed console
    /// line are the same value, so the panel cannot grow a capability the
    /// console lacks. Pinned on the spelling the scene actually submits.
    #[test]
    fn every_edit_the_panel_submits_is_a_line_the_console_command_accepts() {
        for edit in [
            SceneEdit::Insert {
                path: path("root.0"),
                combinator: Combinator::SmoothUnion,
                leaf: LeafKind::Box,
            },
            SceneEdit::Remove {
                path: path("root.1"),
            },
            SceneEdit::Set {
                path: path("root.1"),
                param: Param::Center,
                value: EditValue::Vector(Vec3::new(-0.125, 0.5, 1.0 / 3.0)),
            },
        ] {
            let line = CommandLine {
                name: "sdf".to_string(),
                args: edit.to_args(),
            };
            assert_eq!(
                SceneEdit::from_args(&line.arg_refs()).expect("the console parses it"),
                edit,
            );
        }
    }

    /// A rebuild happens when the tree changed and not otherwise. The compile
    /// is the expensive half of the loop, so "an edit that changed nothing
    /// costs nothing" is the property that keeps a held slider from being a
    /// compile per frame.
    #[test]
    fn the_generation_advances_once_per_landed_edit_and_never_for_a_no_op() {
        let mut editor = Editor {
            scene: boot_scene(),
            generation: 0,
        };
        let edit = SceneEdit::Set {
            path: path("root.1"),
            param: Param::Radius,
            value: EditValue::Scalar(0.2),
        };
        assert!(editor.apply(&edit).expect("applies"));
        assert_eq!(editor.generation, 1);
        assert!(!editor.apply(&edit).expect("applies"));
        assert_eq!(editor.generation, 1);

        assert!(editor
            .apply(&SceneEdit::Remove {
                path: NodePath::root(),
            })
            .is_err());
        assert_eq!(editor.generation, 1, "a refused edit is not a version");
    }
}
