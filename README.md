# ArtExtract

Deterministic C++17 CLI for extracting file-level signals from an input artifact.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/artextract extract --input README.md
./build/artextract extract --input README.md --output report.json
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Output fields

- `bytes_total`: total bytes in input.
- `lines_total`: count of `\n` bytes.
- `printable_bytes`: bytes classified as printable ASCII or whitespace.
- `printable_ratio`: `printable_bytes / bytes_total`.
- `dominant_byte`: most frequent byte value in `[0,255]`.
- `dominant_byte_count`: frequency of `dominant_byte`.
- `hash_fnv1a64_hex`: streaming FNV-1a 64-bit digest (non-cryptographic).
