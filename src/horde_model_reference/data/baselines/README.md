# Image Generation Baselines

`catalog.json` is the packaged bootstrap for the served image baseline catalog. It seeds a PRIMARY
deployment on first start and gives a REPLICA a coherent vocabulary before it has reached PRIMARY.

Baselines are a first-class served resource, not a model category. Edit them through the
`/model_references/v2/image_generation/baselines/change-sets` endpoint and the pending queue; a
deployment's runtime copy under the data root wins over this file.

`capabilities` records architecture and ecosystem facts (whether weights of a given kind exist for
the family at all). Whether a worker's engine can run one of them is a separate axis tracked by the
worker, not here.
