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
  key.rs       — Key trait (Copy + Ord + to_model_input + to_exact_ordinal)
  model.rs     — LinearModel, FMCD fitting algorithm
  node.rs      — Node<K, V> struct, slot types
  build.rs     — bulk-load construction from sorted data
  lookup.rs    — recursive lookup algorithm
  insert.rs    — insert with chain-method conflict resolution
  rebuild.rs   — localized subtree rebuild via CAS-swap
  remove.rs    — logical deletion
  iter.rs      — in-order DFS iterator, range queries
  map.rs       — LearnedMap<K, V> public API
  set.rs       — LearnedSet<K> wrapper
  config.rs    — configuration builder
  error.rs     — error types
```

### Phased Implementation

- **Phase 1 (done)**: Single-threaded. No atomics. Validate FMCD, lookup, insert,
  iterator. All methods take `&mut self`.
- **Phase 2 (done)**: Concurrent. Atomic slots, lock-free reads, CAS-based writes,
  epoch-based reclamation. All methods take `&self`. Auto-rebuild heuristic.
- **Phase 2.5 (done → superseded by Phase 4)**: `RwLock<()>` rebuild guard
  was removed in Phase 4. Localized subtree rebuilds replaced the global lock.
- **Phase 3 (done)**: O(n) in-order iteration (DFS is naturally sorted — removed
  unnecessary sort). Range queries: `range(start..end)`, `first_key_value()`,
  `last_key_value()`, `range_count()`. Model-guided seek for O(depth) range init.
- **Phase 4 (done)**: Depth-triggered localized subtree rebuilds. Insert tracks
  descent depth; when it exceeds `rebuild_depth_threshold` (default 8), the
  inserting thread rebuilds the degraded subtree inline via CAS-swap. Replaced
  Phase 2.5 `RwLock` with fully lock-free operation. Global `rebuild()` remains
  as explicit lock-free compaction API.
- **Phase 5**: Production hardening. Three sub-phases:
  - **5a — Model & efficiency (done, except arena audit)**: True FMCD candidate
    slope iteration (done). Adaptive root sizing — lowered initial rebuild
    threshold to 16 and improved initial root to 64 slots/slope=1.0; ramp-up
    went from ~350x to ~5x vs BTreeMap (done). `size_hint` on `Iter` with
    count hint from `LearnedMap::len()` (done). Per-entry memory overhead
    audit — current heap alloc per slot via `Atomic<SlotInner>` causes poor
    cache locality vs BTreeMap's contiguous nodes; explore arena-based slot
    storage (remaining).
  - **5b — Key generalization & API gaps**: Relax `Key: Copy` to `Key: Clone`
    and support `[u8; N]` fixed-size byte arrays (done — enables UUIDs, hashes,
    and other fixed-width binary keys). `AsRef<[u8]>` for heap-allocated keys
    like `String` is remaining — requires storing keys behind `Arc` or similar
    in `SlotInner` to avoid lifetime issues with epoch-based reclamation.
    Add `entry` API / `get_or_insert` for atomic
    check-and-insert. `clear()` (done) / `drain()` (done). `bulk_load_dedup`
    variant that deduplicates instead of rejecting duplicates (done). `len()`
    approximation documented on all four callsites (done).
  - **5c — Durability & operational readiness**: Rebuild-under-concurrency must
    not silently drop writes — add a retry-on-CAS-failure mode or document the
    write-quiescence requirement. Tombstone compaction for remove-heavy workloads
    (done — per-node `num_tombstones` counter, configurable
    `tombstone_ratio_threshold` (default 0.5), piggybacks on localized subtree
    rebuild). Memory estimation API `allocated_bytes()` (done). Optional `serde`
    serialization behind a feature flag. Documentation, benchmarks tuning,
    crates.io publish.
  - **Phase 4.5 (done)**: Fixed stack overflow for keys with identical f64
    representations (u64 above 2^53, nanosecond timestamps). Added
    `Key::to_exact_ordinal() -> i128` and `LinearModel::binary_split()` for
    degenerate conflict resolution.

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

## Development Workflow

### Before Every Commit

1. `cargo test` — all tests must pass. No committing with known failures.
2. `cargo clippy` — zero warnings. Fix issues, don't suppress without reason.
3. Review the diff (`git diff --staged`) before committing.

### Before Every Push

1. Run the full validation sequence:
   ```bash
   cargo clippy && cargo test
   ```
2. If CI exists, verify it passes before merging or pushing to `main`.
3. Never force-push to `main`.

### Commit Practices

- Each commit should be a single logical change (one bug fix, one feature, one refactor).
- Write commit messages that explain *why*, not just *what*.
- Every bug fix commit must include a regression test.
- Don't commit half-working code to `main` — use a feature branch if work is in progress.

### Branch Strategy

- `main` is always green. All tests pass on every commit.
- Feature branches for multi-commit work. Rebase onto `main` before merging.
- Delete feature branches after merge.

## Repository

- Owner: slush97
- Private repo: github.com/slush97/scry-index
