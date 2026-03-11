# Troubleshooting

## 1) `git clone` fails with RPC/EOF errors
Symptoms:
- `error: RPC failed; curl 18 ...`
- `fatal: early EOF`

Fix:
- Skip clone and use downloader fallback:
```bash
python download_images.py --output-dir images --max-images 5000 --workers 24
```
- Or download metadata CSVs directly:
```bash
mkdir -p nga_data/data
curl -L https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/objects.csv -o nga_data/data/objects.csv
curl -L https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/published_images.csv -o nga_data/data/published_images.csv
```

## 2) `ERROR: missing nga_data/data/published_images.csv`
Cause:
- metadata clone incomplete or path mismatch.

Fix:
- Use `--published-csv` explicitly:
```bash
python download_images.py --published-csv /absolute/path/published_images.csv --output-dir images
```
- Or rely on auto-fetch fallback (default behavior).

## 3) Jupyter: `Kernel does not exist`
Cause:
- stale browser tab/session token after kernel/server interruption.

Fix:
1. Stop all running Jupyter servers.
2. Start a fresh server:
```bash
python -m jupyter notebook
```
3. Open only the newly printed URL.
4. In notebook: `Kernel -> Restart Kernel and Clear Outputs`.

## 4) `DeadKernelError` during `nbconvert --execute`
Cause:
- runtime instability / memory pressure / GPU backend issues.

Fix:
- Use baseline notebook for deterministic execution.
- For advanced pipeline, cap sample size:
```bash
python pipeline_clip_faiss.py --max-images 1200 --out-dir analysis_out
```
- If needed, enforce CPU explicitly:
```bash
FORCE_DEVICE=cpu python pipeline_clip_faiss.py --max-images 1200 --out-dir analysis_out
```

## 5) `Segmentation fault` in `clip/model.py`
Cause:
- `openai/clip` instability on some Python/torch/macOS builds.

Fix:
- Keep default backend (`--clip-backend open_clip`).
- Use OpenAI backend only if verified stable:
```bash
python pipeline_clip_faiss.py --clip-backend openai
```

## 6) `OMP: Error #15 ... libomp already initialized`
Cause:
- OpenMP runtime collision across native deps on macOS.

Fix:
- Use the current pipeline version (includes runtime guard).
- If reproducing in custom scripts:
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
```

## 7) PDF export fails from notebook
### `No module named 'playwright'`
```bash
pip install "nbconvert[webpdf]"
python -m playwright install chromium
```

### LaTeX-based PDF export fails
- Use webpdf path above.
- Fallback: export HTML and print to PDF manually.

## 8) Too few images downloaded
Check:
- row count selected
- `download_failures.tsv`

Command to retry with lower timeout pressure and retries:
```bash
python download_images.py --output-dir images --max-images 30000 --workers 16 --timeout 20 --retries 3
```
