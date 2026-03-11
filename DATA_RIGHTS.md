# DATA_RIGHTS

## Rules
1. Do not redistribute copyrighted full-resolution images without explicit rights.
2. Keep source-level provenance for each sample.
3. Only release assets with verified redistributable status.
4. Support takedown/removal workflows.

## Rights Labels
- `public_domain`
- `licensed`
- `permission_granted`
- `fair_use_research_only`
- `unknown`

## Required Provenance Fields
- `sample_id`
- `source_url`
- `artist`
- `title`
- `year`
- `rights_status`
- `license_name`
- `license_url`
- `permission_reference`
- `retrieved_at_utc`
- `redistributable`
- `local_path`

## Release Gate
Use `scripts/filter_redistributable.py` on provenance CSV before packaging any public dataset.
