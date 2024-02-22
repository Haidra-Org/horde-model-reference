# Horde Model Reference

This package provides some tools to help manage the models which power the [AI-Horde](https://github.com/db0/AI-Horde).

## Reference info (.json files)
For now, the legacy reference format will be availible as before at the [original official repo](https://github.com/Haidra-Org/AI-Horde-image-model-reference) used in the past.

### stable_diffusion.json
You can find a schema for an individual record in the file `stable_diffusion.schema.json` in the root of this repository. Also see `stable_diffusion.example.json` for a small example containing two records will all of the fields populated.
### stable_diffusion.json changes
You can see two records which include entries for every field, and the associated metadata in `stable_diffusion.example.json` at the root of this repository.

Some key takeaways for the new `stable_diffusion.json`:
- The following keys have been removed:
  - `type`
  - `download_all`
  - `available`
  - the sub-key `file_path` under `download`
  - the entire key `files` under `config` has been removed.
     - (`config` still contains a `download` key, which is a list of all files to download.)
- `baseline`'s old values have been normalized. The currently valid values are as follows:
  - `stable_diffusion_1`
  - `stable_diffusion_2_768`
  - `stable_diffusion_2_512`
  - `stable_diffusion_xl`
  - `stable_cascade`
- An MD5 sum is no longer included. All models (of all types) will have an SHA included from now on.
- `download` entries optionally contain a new key, `known_slow_download`, which indicates this download host is known to be slow at times.

Moving forward, you can expect the schema to honor at least the existing values. There is a strong possibility additional fields will be added.

## Python library
This repo is also a python library designed to help you integrate the scheme the AI-Horde project uses to manage its models into your project.
### General info
You can install this module through pip:
```
python -m pip install horde_model_reference
```
This library has a number of python classes which may assist you in working with the model reference. The following files may be of interest:
- horde_model_reference\model_reference_records.py
  - Contains pydantic definitions, and some meta information, for all record types.
- horde_model_reference\meta_consts.py
  - Contains many commonly used strings, enums, and certain useful dict lookups.
- horde_model_reference\path_consts.py
  - Contains certain potentially useful paths, path constructors and folder/file name information relevant to the package.

Note that a number of useful imports have been made availible at the `horde_model_reference` import level.

## Horde Moderators/Support Staff

### Validating
When making changes to stable_diffusion.json, you can now validate, format, and standardize it for consistency. You can do this by invoking the following:
```
validate_sd stable_diffusion.json
```
This will give you a success message if the file is standardized. If it is not, you can invoke the following:
```
validate_sd stable_diffusion.json --write validated_and_formatted.json
```
This will write the appropriately normalized json out to the path specified by `--write`. It will only write a file out if the input file is valid json and conforms to the established schema.
