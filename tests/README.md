# tests

Current test coverage:
1. `test_multispectral_contract.py`
   - manifest parser + schema behavior
   - loader tensor shape and channel mask correctness
2. `test_multispectral_model.py`
   - forward output shapes for multitask model
   - gradient-flow sanity check
3. `test_multispectral_cli_smoke.py`
   - end-to-end smoke test for `scripts/train.py` + `scripts/eval.py`
   - validates required report artifacts

Run:
```bash
python -m unittest discover -s tests -p 'test_*.py'
```
