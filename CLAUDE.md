# scry-index

A concurrent sorted key-value map backed by learned index structures (LIPP/SALI).

## What This Project Is

The first Rust implementation of a concurrent learned index. All existing concurrent
learned indexes (SALI, FINEdex, XIndex, APEX) are C++ only. This crate provides a
thread-safe sorted map where multiple threads can read and write simultaneously without
a global lock, using piecewise linear models to predict key positions for O(1) expected
lookup time.

## Architecture

Based on LIPP's "chain method" (Mod.+C strategy) from the SALI paper (SIGMOD 2024):

- **Nodes** contain a linear model (`slope`, `intercept`) and a fixed-size slot array
- **Slots** are one of: `Empty`, `Data(K, V)`, or `Child(Box<Node>)`
- **Lookups** predict a slot index via the model, then follow child pointers on conflict
- **Inserts** on collision create a child node containing both keys (no data shifting)
- **No prediction error**: keys always land at their model-predicted slot; conflicts
  create children, not search ranges

### Module Layout

```
src/
  lib.rs       — crate root, re-exports
  key.rs       — Key trait (Copy + Ord + to_model_input)
  model.rs     — LinearModel, FMCD fitting algorithm
  node.rs      — Node<K, V> struct, slot types
  build.rs     — bulk-load construction from sorted data
  lookup.rs    — recursive lookup algorithm
  insert.rs    — insert with chain-method conflict resolution
  remove.rs    — logical deletion
  iter.rs      — in-order DFS iterator, range queries
  map.rs       — LearnedMap<K, V> public API
  set.rs       — LearnedSet<K> wrapper
  config.rs    — configuration builder
  error.rs     — error types
```

### Phased Implementation

- **Phase 1 (current)**: Single-threaded. No atomics. Validate FMCD, lookup, insert,
  iterator. All methods take `&mut self`.
- **Phase 2**: Concurrent. Atomic slots, lock-free reads, per-slot write locks,
  epoch-based reclamation. All methods take `&self`.
- **Phase 3**: SALI node evolving strategies (hot/cold nodes, adaptive rebuilding).
- **Phase 4**: Polish, range queries, benchmarks, crates.io publish.

## Code Conventions

### Absolute Rules

- **No `unsafe` code.** `#[forbid(unsafe_code)]` is set in Cargo.toml. Phase 2
  concurrency will use safe abstractions from crossbeam. If we hit a wall, we
  discuss and explicitly opt in per-module, never globally.
- **Clippy pedantic + nursery.** All warnings are errors in CI. Fix them, don't
  suppress them, unless there is a documented reason in the allow list.
- **Every public item has a doc comment.** `missing_docs = "warn"` is enforced.
- **No panics in library code.** Return `Result` or `Option`. Panics are bugs.
  `unwrap()` is only acceptable in tests.

### Testing Requirements

- **Unit tests** in each module (`#[cfg(test)] mod tests`).
- **Property-based tests** using `proptest` or manual randomized tests for
  core algorithms (FMCD fitting, insert/lookup roundtrip).
- **Integration tests** in `tests/` for end-to-end behavior.
- **Benchmarks** in `benches/` using criterion, comparing against `BTreeMap`.
- **Every bug fix must add a regression test.**
- Target: >90% line coverage on core modules (model, node, lookup, insert).

### Style

- Prefer iterators over index loops.
- Prefer `let` bindings over long method chains (readability over cleverness).
- Group imports: std, external crates, crate-internal. Separate with blank lines.
- Error types use `#[non_exhaustive]` for forward compatibility.
- No `println!` or `dbg!` in library code. Use `log` crate if tracing is needed.

## Key References

- **SALI paper**: "Scalable Adaptive Learned Index Framework" (SIGMOD 2024)
  https://people.iiis.tsinghua.edu.cn/~huanchen/publications/sali-sigmod24.pdf
- **LIPP paper**: "Updatable Learned Index with Precise Positions" (PVLDB 2021)
- **SALI C++ reference**: https://github.com/cds-ruc/SALI (~4k LOC, single header)
- **PGM++ paper**: "Why Are Learned Indexes So Effective but Sometimes Ineffective?"
  (PVLDB 2025) — identifies correction scan as the real bottleneck
- **crossbeam-epoch docs**: https://docs.rs/crossbeam/latest/crossbeam/epoch/

## Build and Test

```bash
cargo test                        # run all tests
cargo test -- --nocapture         # with output
cargo clippy                      # lint check
cargo bench                       # benchmarks
```

## Repository

- Owner: slush97
- Private repo: github.com/slush97/scry-index
- Branch strategy: `main` is always green. Feature branches for phases.
