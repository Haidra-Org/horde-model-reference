## Foreword
These are not all 1:1 matches to the legacy files. If there is a need to update these, please look carefully at the diffs. 4-indent format any json files first.

## To convert files
- `convert_legacy.py` is preconfigured to work with the files in `legacy/`. The resulting files will overwrite any existing reference dbs in `horde_model_reference/` (or whatever the code root of this project currently is).
- Invoke `python convert_legacy.py`
- Take note of any errors. If you get a raw stack trace and/or a message to stdout with "CRITICAL:", confirm the data is the correct schema. Pydantic is fairly verbose, a careful read of the stack trace will likely reveal the problem.
- Check the relevant `.log` files for any non-critical, but potentially problematic, issues
- Check for any missing sha hashes in the jsons - `FIXME` appears as a default.
- Enjoy
