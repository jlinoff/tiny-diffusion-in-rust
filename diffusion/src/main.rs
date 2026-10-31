//! `diffusion`: the command line. Each stage of PLAN.md adds its commands;
//! for now there is one, `info`, which shows what the build contains.

use clap::{Parser, Subcommand};
use mini_diffusion::blas;

#[derive(Parser)]
#[command(
    name = "diffusion",
    version,
    about = "A small diffusion model, built by hand (see PLAN.md)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show what this build contains.
    Info,
}

fn main() {
    match Cli::parse().command {
        Command::Info => {
            println!("diffusion {}", env!("CARGO_PKG_VERSION"));
            println!("  large matrix products: {}", blas::backend());
            println!("  stages built: foundation (PLAN.md, Order of work)");
        }
    }
}
