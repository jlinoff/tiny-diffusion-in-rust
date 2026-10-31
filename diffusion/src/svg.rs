//! Plots as SVG, the text format for drawings that browsers (and GitHub)
//! display: points, paths and arrows in a square window of the plane, side
//! by side in panels with titles. No library; a figure is a string.

use crate::tensor::Mat;
use std::fmt::Write;

/// The plane's window, the same for every panel: [−LIM, LIM] on both axes.
pub const LIM: f32 = 3.0;
/// Each panel's size in pixels, and the gap above it for the title.
const SIZE: f32 = 320.0;
const TITLE: f32 = 28.0;
const GAP: f32 = 12.0;
/// The most dots drawn per call to `points`.
pub const MAX_POINTS: usize = 1000;

/// One panel: a title and the shapes drawn in it.
pub struct Panel {
    title: String,
    body: String,
}

/// Plane coordinates to pixels within a panel (y grows upwards in the
/// plane, downwards in SVG).
fn px(x: f32, y: f32) -> (f32, f32) {
    (
        (x + LIM) / (2.0 * LIM) * SIZE,
        (LIM - y) / (2.0 * LIM) * SIZE,
    )
}

fn inside(x: f32, y: f32) -> bool {
    x.abs() <= LIM && y.abs() <= LIM
}

impl Panel {
    pub fn new(title: &str) -> Panel {
        Panel {
            title: title.into(),
            body: String::new(),
        }
    }

    /// Each row of `m` (n × 2) as a dot, the first `MAX_POINTS` at most (a
    /// figure stays small enough to commit); points outside the window are
    /// left out.
    pub fn points(&mut self, m: &Mat<f32>, color: &str, radius: f32, opacity: f32) {
        let _ = write!(self.body, r#"<g fill="{color}" fill-opacity="{opacity}">"#);
        for r in 0..m.rows.min(MAX_POINTS) {
            let (x, y) = (m.row(r)[0], m.row(r)[1]);
            if inside(x, y) {
                let (a, b) = px(x, y);
                let _ = write!(
                    self.body,
                    r#"<circle cx="{a:.0}" cy="{b:.0}" r="{radius}"/>"#
                );
            }
        }
        self.body.push_str("</g>");
    }

    /// Paths: `steps[k]` holds every point after step k; point r's path
    /// joins row r of each, for the first `count` points.
    pub fn paths(&mut self, steps: &[Mat<f32>], count: usize, color: &str) {
        let n = steps.first().map_or(0, |m| m.rows.min(count));
        let _ = write!(
            self.body,
            r#"<g fill="none" stroke="{color}" stroke-width="0.8" stroke-opacity="0.6">"#
        );
        for r in 0..n {
            let pts: Vec<String> = steps
                .iter()
                .map(|m| {
                    let (x, y) = (m.row(r)[0].clamp(-LIM, LIM), m.row(r)[1].clamp(-LIM, LIM));
                    let (a, b) = px(x, y);
                    format!("{a:.0},{b:.0}")
                })
                .collect();
            let _ = write!(self.body, r#"<polyline points="{}"/>"#, pts.join(" "));
        }
        self.body.push_str("</g>");
    }

    /// Arrows from each row of `at` along the matching row of `v`, scaled by
    /// `scale` (plane units per unit of velocity).
    pub fn arrows(&mut self, at: &Mat<f32>, v: &Mat<f32>, scale: f32, color: &str) {
        for r in 0..at.rows {
            let (x, y) = (at.row(r)[0], at.row(r)[1]);
            let (dx, dy) = (v.row(r)[0] * scale, v.row(r)[1] * scale);
            let (a, b) = px(x, y);
            let (c, d) = px(x + dx, y + dy);
            let _ = write!(
                self.body,
                r#"<line x1="{a:.1}" y1="{b:.1}" x2="{c:.1}" y2="{d:.1}" stroke="{color}" stroke-width="1" marker-end="url(#head)"/>"#
            );
        }
    }
}

/// A grid of points covering the window, `n` × `n`, as rows of a matrix:
/// where to draw a velocity field's arrows.
pub fn grid(n: usize) -> Mat<f32> {
    let mut data = Vec::with_capacity(2 * n * n);
    for i in 0..n {
        for j in 0..n {
            let f = |k: usize| -LIM + (k as f32 + 0.5) * 2.0 * LIM / n as f32;
            data.push(f(j));
            data.push(f(i));
        }
    }
    Mat::from_vec(n * n, 2, data)
}

/// Panels in rows of `cols`, under an overall caption.
pub fn figure(caption: &str, panels: &[Panel], cols: usize) -> String {
    let cols = cols.max(1).min(panels.len().max(1));
    let rows = panels.len().div_ceil(cols);
    let w = cols as f32 * (SIZE + GAP) + GAP;
    let h = rows as f32 * (SIZE + TITLE + GAP) + 40.0;
    let mut s = String::new();
    let _ = write!(
        s,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="sans-serif">"#
    );
    s.push_str(r#"<defs><marker id="head" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="context-stroke"/></marker></defs>"#);
    let _ = write!(s, r#"<rect width="{w}" height="{h}" fill="white"/>"#);
    let _ = write!(
        s,
        r#"<text x="{GAP}" y="26" font-size="16" font-weight="bold">{}</text>"#,
        escape(caption)
    );
    for (k, p) in panels.iter().enumerate() {
        let (c, r) = (k % cols, k / cols);
        let x = GAP + c as f32 * (SIZE + GAP);
        let y = 40.0 + r as f32 * (SIZE + TITLE + GAP);
        let _ = write!(
            s,
            r#"<text x="{x}" y="{:.1}" font-size="13">{}</text>"#,
            y + 18.0,
            escape(&p.title)
        );
        let _ = write!(
            s,
            r##"<g transform="translate({x},{:.1})"><rect width="{SIZE}" height="{SIZE}" fill="#fafafa" stroke="#ccc"/>{}</g>"##,
            y + TITLE,
            p.body
        );
    }
    s.push_str("</svg>\n");
    s
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_figure_holds_its_shapes_and_titles() {
        let mut p = Panel::new("dots & <paths>");
        p.points(
            &Mat::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 9.0, 9.0]),
            "red",
            2.0,
            1.0,
        );
        p.paths(
            &[
                Mat::zeros(2, 2),
                Mat::from_vec(2, 2, vec![1.0, 1.0, -1.0, -1.0]),
            ],
            5,
            "blue",
        );
        p.arrows(&grid(2), &Mat::from_vec(4, 2, vec![1.0; 8]), 0.5, "gray");
        let s = figure("caption", &[p, Panel::new("empty")], 2);
        assert!(s.starts_with("<svg") && s.ends_with("</svg>\n"));
        assert_eq!(s.matches("<circle").count(), 2); // (9, 9) is outside the window
        assert_eq!(s.matches("<polyline").count(), 2);
        assert_eq!(s.matches("<line ").count(), 4);
        assert!(s.contains("dots &amp; &lt;paths&gt;"));
        assert_eq!(s.matches("<g transform").count(), 2);
    }

    #[test]
    fn the_window_maps_onto_the_panel() {
        assert_eq!(px(-LIM, LIM), (0.0, 0.0));
        assert_eq!(px(LIM, -LIM), (SIZE, SIZE));
        assert_eq!(px(0.0, 0.0), (SIZE / 2.0, SIZE / 2.0));
    }

    #[test]
    fn grid_covers_the_window_evenly() {
        let g = grid(3);
        assert_eq!(g.rows, 9);
        assert_eq!(g.row(0), &[-2.0, -2.0]);
        assert_eq!(g.row(4), &[0.0, 0.0]);
        assert_eq!(g.row(8), &[2.0, 2.0]);
    }
}
