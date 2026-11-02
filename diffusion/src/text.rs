//! Text for stage 6, read from tiny-llm-in-rust's files, so the two models
//! are trained and judged on exactly the same tokens (PLAN.md, stage 6):
//!
//! - the token-ID file written by its tokenizer (`out/corpus.ids`), with
//!   the same held-out split and the same random windows;
//! - the tokenizer file (`out/tok.bpe`), for decoding IDs back to text.
//!
//! Both formats, and the split and window rules, follow tiny-llm-in-rust's
//! `transformer/src/data.rs` and `tokenizer.rs` (copied in behaviour, not in
//! code: nothing here encodes text or checks fingerprints).

use crate::rng::Rng;
use std::path::Path;

/// A token-ID file: a 64-byte header (magic b"MBPEIDS\0", version 1,
/// bytes per ID 2 or 4, V, reserved, the count, a 32-byte fingerprint),
/// then the IDs, little-endian.
pub struct Corpus {
    pub ids: Vec<u32>,
    pub vocab_size: usize,
}

impl Corpus {
    pub fn load(path: &Path) -> Result<Corpus, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
        Self::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn decode(bytes: &[u8]) -> Result<Corpus, String> {
        if bytes.len() < 64 {
            return Err("too short for the 64-byte header".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        if &bytes[..8] != b"MBPEIDS\0" || u32_at(8) != 1 {
            return Err("not a mini_bpe token-ID file (version 1)".into());
        }
        let (width, vocab_size) = (u32_at(12) as usize, u32_at(16) as usize);
        let count = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        if width != 2 && width != 4 {
            return Err(format!("{width} bytes per ID (expected 2 or 4)"));
        }
        let body = &bytes[64..];
        if body.len() != count * width {
            return Err(format!(
                "header says {count} IDs of {width} bytes, file holds {} bytes",
                body.len()
            ));
        }
        let ids: Vec<u32> = if width == 2 {
            body.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                .collect()
        } else {
            body.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        if let Some(&bad) = ids.iter().find(|&&t| t as usize >= vocab_size) {
            return Err(format!("token ID {bad} is not below V = {vocab_size}"));
        }
        Ok(Corpus { ids, vocab_size })
    }
}

/// The held-out split, as tiny-llm-in-rust makes it: the last
/// round(len × fraction) tokens are for validation.
pub fn split(ids: &[u32], valid_fraction: f64) -> (&[u32], &[u32]) {
    let n_valid = ((ids.len() as f64) * valid_fraction).round() as usize;
    ids.split_at(ids.len() - n_valid)
}

/// The training windows of a step, as tiny-llm-in-rust draws them with
/// random sampling: `batch` starts, each `rng.below(len − seq)` from the
/// stream for (seed, step). tiny-llm-in-rust's windows are seq + 1 tokens
/// (inputs and next-token targets); a masked model needs only the first
/// seq, the same text.
pub fn window_starts(len: usize, batch: usize, seq: usize, seed: u64, step: u64) -> Vec<usize> {
    let mut rng = Rng::stream(seed, step);
    let span = len - seq;
    (0..batch).map(|_| rng.below(span)).collect()
}

/// Non-overlapping windows of `seq` tokens from the start of `data`, at
/// most `max`: the held-out windows the diffusion bound is measured on.
pub fn valid_windows(data: &[u32], seq: usize, max: usize) -> Vec<&[u32]> {
    data.chunks_exact(seq).take(max).collect()
}

/// A tokenizer file (`tok.bpe`), enough to decode: each token's bytes.
/// Lines: '#' comments, one "pattern …" line, one "<left> <right>" line
/// per merge (merge k makes token 256 + k), one "special <name>" line per
/// special token (IDs after the merges). V = 256 + merges + specials.
pub struct Vocab {
    pieces: Vec<Vec<u8>>,
    pub specials: Vec<String>,
    pub merges: usize,
}

impl Vocab {
    pub fn load(path: &Path) -> Result<Vocab, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("can't read tokenizer {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("tokenizer {}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Vocab, String> {
        let mut pieces: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        let mut specials = Vec::new();
        let mut merges = 0;
        for (i, line) in text.lines().enumerate() {
            if line.starts_with("pattern ") {
                continue;
            }
            let data: Vec<&str> = line.split('#').next().unwrap().split_whitespace().collect();
            match data.as_slice() {
                [] => {}
                ["special", name] => specials.push(name.to_string()),
                [a, b] => {
                    let next = pieces.len();
                    let id = |s: &str| {
                        s.parse::<usize>()
                            .ok()
                            .filter(|&v| v < next)
                            .ok_or(format!("line {}: token id {s} out of range", i + 1))
                    };
                    let mut piece = pieces[id(a)?].clone();
                    piece.extend_from_slice(&pieces[id(b)?]);
                    pieces.push(piece);
                    merges += 1;
                }
                _ => return Err(format!("line {}: expected '<left_id> <right_id>'", i + 1)),
            }
        }
        for name in &specials {
            pieces.push(name.as_bytes().to_vec());
        }
        Ok(Vocab {
            pieces,
            specials,
            merges,
        })
    }

    pub fn size(&self) -> usize {
        self.pieces.len()
    }

    /// The ID of a special token, such as "<|endoftext|>".
    pub fn special(&self, name: &str) -> Option<usize> {
        self.specials
            .iter()
            .position(|s| s == name)
            .map(|i| 256 + self.merges + i)
    }

    /// IDs to text; a cut-off UTF-8 character shows as the replacement
    /// character. Any ID at or past V (the model's MASK) shows as "▒".
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &i in ids {
            match self.pieces.get(i) {
                Some(p) => bytes.extend_from_slice(p),
                None => bytes.extend_from_slice("▒".as_bytes()),
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tiny-llm-in-rust's fixtures: tiny.bpe (V = 317, one special) and
    /// tiny.ids (2,356 IDs, which begin "The" and end with the special).
    #[test]
    fn reads_tiny_llm_fixtures() {
        let v = Vocab::parse(include_str!("../tests/data/tiny.bpe")).unwrap();
        assert_eq!((v.size(), v.merges), (317, 60));
        assert_eq!(v.special("<|endoftext|>"), Some(316));
        let c = Corpus::decode(include_bytes!("../tests/data/tiny.ids")).unwrap();
        assert_eq!((c.ids.len(), c.vocab_size), (2356, 317));
        let first: Vec<usize> = c.ids[..12].iter().map(|&t| t as usize).collect();
        assert_eq!(
            first,
            vec![84, 104, 101, 272, 287, 266, 287, 301, 264, 280, 287, 46]
        );
        assert!(v.decode(&first).starts_with("The"));
        assert_eq!(*c.ids.last().unwrap(), 316);
        assert!(v.decode(&[316]).contains("endoftext"));
        assert_eq!(v.decode(&[317]), "▒");
    }

    #[test]
    fn rejects_bad_files() {
        assert!(Corpus::decode(b"short").is_err());
        let mut bytes = include_bytes!("../tests/data/tiny.ids").to_vec();
        bytes[0] = b'X';
        assert!(Corpus::decode(&bytes).is_err());
        assert!(Vocab::parse("pattern x\n5 999\n").is_err());
    }

    /// The split and the windows follow tiny-llm-in-rust's rules.
    #[test]
    fn split_and_windows() {
        let ids: Vec<u32> = (0..1000).collect();
        let (train, valid) = split(&ids, 0.05);
        assert_eq!((train.len(), valid.len(), valid[0]), (950, 50, 950));
        let s = window_starts(train.len(), 8, 256, 1, 3);
        assert_eq!(s.len(), 8);
        assert!(s.iter().all(|&x| x < 950 - 256));
        assert_eq!(s, window_starts(train.len(), 8, 256, 1, 3));
        assert_ne!(s, window_starts(train.len(), 8, 256, 1, 4));
        assert_eq!(valid_windows(&ids, 256, 10).len(), 3);
    }
}
