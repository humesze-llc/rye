![Letters tumbling into the Loam wordmark](assets/readme/hero.webp)
Loam is an engine for exploring geometry beyond ordinary 3D space!

[![CI](https://github.com/throgsoft/loam/actions/workflows/ci.yml/badge.svg)](https://github.com/throgsoft/loam/actions/workflows/ci.yml)

## Goals

I'm building Loam to provide tooling to build games where space is a gimmick. Higher dimensions, curved space, portals, weird topology, and compelling visual effects.

## Polytope Playground

Rotate 4D shapes and watch their 3D cross-sections change.

![Regular 4-polytopes turning through a rotation plane while their 3D cross-sections change](assets/readme/rotate.webp)

Or pick them up and throw them around the 4D Toybox, directly inspired by [Marc ten Bosch](https://marctenbosch.com/)'s [4D Toys](https://4dtoys.com/).

![Polychora dropped into a box under 4D gravity, picked up and thrown](assets/readme/toybox.webp)

## Play

Install Rust with [rustup](https://rust-lang.org/tools/install/). On Windows, run the installer and accept the C++ build tools prompt. On macOS or Linux:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Open a new terminal after installation. On macOS, install the command line tools with `xcode-select --install`.

<details>
<summary>Ubuntu / Debian build dependencies</summary>

```sh
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libwayland-dev \
  libxkbcommon-dev libxkbcommon-x11-dev libx11-dev libxi-dev \
  libxcursor-dev libxrandr-dev
```

</details>

Clone the repository, then build and run the playground. Rustup installs the toolchain pinned by the project:

```sh
git clone https://github.com/throgsoft/loam.git
cd loam
cargo run --release --locked -p polytope_playground
```

Use the Demo menu to switch scenes. Space pauses rotation. Backtick opens the console.

For the browser build:

```sh
rustup target add wasm32-unknown-unknown
cargo install --locked trunk
trunk serve crates/polytope_playground/index.html
```

[API documentation](https://throgsoft.github.io/loam/)

## License

MIT OR Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
