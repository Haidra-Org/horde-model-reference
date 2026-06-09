# Horde Model Reference

<p align="center">
  <img src="docs/haidra-assets/haidra-logo/haidra-transparent.webp" alt="Haidra Logo" width="50"/>
</p>

<p align="center">
  <a href="https://github.com/Haidra-Org/horde-model-reference/actions"><img src="https://github.com/Haidra-Org/horde-model-reference/workflows/Tests/badge.svg" alt="Build Status"></a>
  <a href="https://pypi.org/project/horde-model-reference/"><img src="https://img.shields.io/pypi/v/horde-model-reference.svg" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/horde-model-reference/"><img src="https://img.shields.io/pypi/pyversions/horde-model-reference.svg" alt="Supported Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" alt="License: AGPL v3"></a>
</p>

**Horde Model Reference** is the authoritative source for AI model metadata in the [AI-Horde](https://aihorde.net) ecosystem: download URLs, SHA-256 checksums, baselines, NSFW flags, and capabilities for image generation, text generation, and utility models used by workers, clients, and services.

It is three things in one package: a set of JSON reference files, a Python library, and a FastAPI service.

## Install

```bash
pip install horde-model-reference
```

## Quick try

```python
from horde_model_reference import ModelReferenceManager, MODEL_REFERENCE_CATEGORY

manager = ModelReferenceManager()  # auto-fetches from the PRIMARY server, caches in memory

models = manager.get_model_reference(MODEL_REFERENCE_CATEGORY.image_generation)
for name, model in models.items():
    print(f"{name}: {model.baseline}, NSFW={model.nsfw}")
```

No configuration needed -- the defaults work out of the box. The library runs in read-only REPLICA mode, fetching from `models.aihorde.net` with automatic fallback to the original GitHub repositories if the PRIMARY server is unreachable.

For the fluent query API (filter, sort, aggregate, merge sources), HTTP API access, deploying your own server, or contributing model data, see the [full documentation](https://horde-sdk.readthedocs.io/en/latest/).

## Documentation

Full docs at **[horde-model-reference.readthedocs.io/en/latest/](https://horde-model-reference.readthedocs.io/en/latest/)**. Build locally with `mkdocs serve`.

- [Getting Started](https://horde-model-reference.readthedocs.io/en/latest/tutorials/getting_started/) -- first query, singleton pattern, prefetch strategies
- [Querying Models](https://horde-model-reference.readthedocs.io/en/latest/tutorials/querying_models/) -- filter, sort, aggregate with the fluent API
- [Using the HTTP API](https://horde-model-reference.readthedocs.io/en/latest/tutorials/using_the_http_api/) -- call the live service over HTTP
- [Filter Models for a Worker](https://horde-model-reference.readthedocs.io/en/latest/guides/filter_models_for_a_worker/) -- narrow the reference to what your node should serve
- [Configuration & Troubleshooting](https://horde-model-reference.readthedocs.io/en/latest/tutorials/configuration_and_troubleshooting/) -- env vars, debugging, common issues
- [Deployment Guide](DEPLOYMENT.md) -- run your own PRIMARY server

Runnable examples live in the [`examples/`](examples/) directory.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup and guidelines, and see the [open issues](https://github.com/Haidra-Org/horde-model-reference/issues).

## Community

- **Discord**: [AI Horde Discord](https://discord.gg/3DxrhksKzn) -- `#horde-model-reference` channel
- **Bug reports**: [GitHub Issues](https://github.com/Haidra-Org/horde-model-reference/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/Haidra-Org/horde-model-reference/discussions)

## License

Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for the full text. If you run a modified version as a network service, you must make your source code available to users. See the [GNU AGPL FAQ](https://www.gnu.org/licenses/agpl-3.0.html) for details.

## Related Projects

- **[AI-Horde](https://github.com/Haidra-Org/AI-Horde)** -- main AI-Horde API server
- **[horde-worker-reGen](https://github.com/Haidra-Org/horde-worker-reGen)** -- official image generation worker
- **[horde-sdk](https://github.com/Haidra-Org/horde-sdk)** -- Python SDK for the AI-Horde API
- **[hordelib](https://github.com/Haidra-Org/hordelib)** -- library wrapper around ComfyUI
