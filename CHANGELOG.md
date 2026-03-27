# Changelog

## 0.1.0 — 2026-03-26

Initial release.

### Features

- `LearnedMap<K, V>` — concurrent sorted map with O(1) expected lookups
- `LearnedSet<K>` — sorted set wrapper
- Lock-free reads via epoch-based reclamation (crossbeam)
- CAS-based writes with per-slot contention
- FMCD model fitting for zero-conflict bulk loading
- Sorted iteration and range queries (`range`, `first_key_value`, `last_key_value`)
- Depth-triggered localized subtree rebuilds
- Tombstone compaction for remove-heavy workloads
- `get_or_insert` / `get_or_insert_with` atomic entry API
- `clear()` / `drain()` for bulk removal
- `allocated_bytes()` memory introspection
- `bulk_load` and `bulk_load_dedup` for fast construction from sorted data
- `MapRef` / `SetRef` convenience handles (bundles map + epoch guard)
- Key types: integers, `[u8; N]`, `String`, `Vec<u8>`
- Optional serde support (`features = ["serde"]`)
- Configurable expansion factor, rebuild thresholds, and tombstone compaction
