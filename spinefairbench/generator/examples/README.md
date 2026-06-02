# Generator Examples

No clinical source radiographs are included in git.

Use the dry-run smoke script to create a synthetic non-clinical input and
metadata bundle:

```bash
python3 scripts/run_generator_smoke_test.py --dry-run --output /tmp/spinefairbench_generator_smoke
```

The generated synthetic image is only for code-path validation. It is not a
clinical image and is not part of the SpineFairBench benchmark.
