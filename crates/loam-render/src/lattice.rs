//! A [`Viewport`] is a rectangle in pixel coordinates (origin top-left, +y down).

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Viewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    pub fn full(framebuffer: [u32; 2]) -> Self {
        Self {
            x: 0,
            y: 0,
            width: framebuffer[0],
            height: framebuffer[1],
        }
    }

    /// Empty when the panel covers the framebuffer.
    pub fn right_of_left_panel(panel_width: u32, framebuffer: [u32; 2]) -> Self {
        let panel = panel_width.min(framebuffer[0]);
        Self {
            x: panel,
            y: 0,
            width: framebuffer[0] - panel,
            height: framebuffer[1],
        }
    }

    /// The depth range is set to the standard `[0.0, 1.0]`.
    pub fn apply(&self, rp: &mut wgpu::RenderPass<'_>) {
        rp.set_viewport(
            self.x as f32,
            self.y as f32,
            self.width as f32,
            self.height as f32,
            0.0,
            1.0,
        );
    }

    /// The layout the hyperslice kernel's `u.resolution` uniform expects.
    pub fn resolution_f32(&self) -> [f32; 2] {
        [self.width as f32, self.height as f32]
    }

    /// Cells are `width / n` pixels; the trailing cell absorbs the rounding
    /// remainder so the strip covers `self` without seams. Empty when `n == 0`.
    pub fn split_horizontal(&self, n: u32) -> Vec<Viewport> {
        if n == 0 {
            return Vec::new();
        }
        let cell_w = self.width / n;
        let remainder = self.width - cell_w * n;
        (0..n)
            .map(|i| {
                let extra = if i == n - 1 { remainder } else { 0 };
                Viewport {
                    x: self.x + i * cell_w,
                    y: self.y,
                    width: cell_w + extra,
                    height: self.height,
                }
            })
            .collect()
    }

    /// Same trailing-cell remainder rule as [`Self::split_horizontal`].
    pub fn split_vertical(&self, n: u32) -> Vec<Viewport> {
        if n == 0 {
            return Vec::new();
        }
        let cell_h = self.height / n;
        let remainder = self.height - cell_h * n;
        (0..n)
            .map(|i| {
                let extra = if i == n - 1 { remainder } else { 0 };
                Viewport {
                    x: self.x,
                    y: self.y + i * cell_h,
                    width: self.width,
                    height: cell_h + extra,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn right_of_left_panel_carves_correctly() {
        let v = Viewport::right_of_left_panel(300, [1280, 720]);
        assert_eq!(v.x, 300);
        assert_eq!(v.y, 0);
        assert_eq!(v.width, 980);
        assert_eq!(v.height, 720);
    }

    #[test]
    fn right_of_left_panel_clamps_when_panel_exceeds_framebuffer() {
        let v = Viewport::right_of_left_panel(2000, [1280, 720]);
        assert_eq!(v.x, 1280);
        assert_eq!(v.width, 0);
    }

    #[test]
    fn split_horizontal_tiles_without_gaps() {
        // 17 doesn't divide evenly into 5; verify remainder lands on the last cell.
        let v = Viewport {
            x: 100,
            y: 50,
            width: 17,
            height: 30,
        };
        let cells = v.split_horizontal(5);
        assert_eq!(cells.len(), 5);
        assert_eq!(cells.first().unwrap().x, 100);
        assert_eq!(
            cells.last().unwrap().x + cells.last().unwrap().width,
            100 + 17
        );
        for win in cells.windows(2) {
            assert_eq!(win[0].x + win[0].width, win[1].x);
        }
        assert_eq!(cells[0].width, 3);
        assert_eq!(cells[4].width, 5);
    }

    #[test]
    fn split_horizontal_n_zero_returns_empty() {
        let v = Viewport::full([100, 50]);
        assert!(v.split_horizontal(0).is_empty());
    }
}
