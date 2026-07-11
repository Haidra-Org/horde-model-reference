"""Image-generation-specific model constants, enums, and descriptors."""

from dataclasses import dataclass, field
from enum import auto

from strenum import StrEnum

from horde_model_reference.registries import DescriptorRegistry, EnumRegistry


class KNOWN_IMAGE_GENERATION_BASELINE(StrEnum):
    """An enum of all the image generation baselines."""

    infer = auto()
    """The baseline is not known and should be inferred from the model name."""

    stable_diffusion_1 = auto()
    stable_diffusion_2_768 = auto()
    stable_diffusion_2_512 = auto()
    stable_diffusion_xl = auto()
    stable_cascade = auto()
    flux_1 = auto()  # TODO: Extract flux and create "IMAGE_GENERATION_BASELINE_CATEGORY" due to name inconsistency
    flux_schnell = auto()  # FIXME
    flux_dev = auto()  # FIXME
    qwen_image = auto()
    z_image_turbo = auto()


@dataclass(frozen=True)
class BaselineDescriptor:
    """Describes a known image-generation baseline in a single place.

    Attributes:
        native_resolution: Preferred single-side resolution, or ``None`` for baselines
            like ``infer`` that have no fixed resolution.
        alternative_names: Alternative human/API names that map to this baseline.

    """

    native_resolution: int | None
    alternative_names: tuple[str, ...] = field(default_factory=tuple)


IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP: dict[KNOWN_IMAGE_GENERATION_BASELINE | str, int] = {}
"""The single-side preferred resolution for each known stable diffusion baseline."""

_ALTERNATIVE_NAME_TO_BASELINE: dict[str, KNOWN_IMAGE_GENERATION_BASELINE | str] = {}


def _rebuild_baseline_derived_data(
    data: dict[KNOWN_IMAGE_GENERATION_BASELINE | str, BaselineDescriptor],
) -> None:
    """Rebuild derived baseline lookups from the registry."""
    IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP.clear()
    IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP.update(
        {b: d.native_resolution for b, d in data.items() if d.native_resolution is not None}
    )

    _ALTERNATIVE_NAME_TO_BASELINE.clear()
    for bl, desc in data.items():
        for alt in desc.alternative_names:
            _ALTERNATIVE_NAME_TO_BASELINE[alt] = bl


_IMAGE_BASELINE_REGISTRY = DescriptorRegistry[KNOWN_IMAGE_GENERATION_BASELINE | str, BaselineDescriptor](
    _rebuild_baseline_derived_data
)


def register_image_baseline(name: KNOWN_IMAGE_GENERATION_BASELINE | str, descriptor: BaselineDescriptor) -> None:
    """Register a new image-generation baseline."""
    _IMAGE_BASELINE_REGISTRY.register(name, descriptor)


register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.infer,
    BaselineDescriptor(native_resolution=None),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_1,
    BaselineDescriptor(
        native_resolution=512,
        alternative_names=(
            "stable diffusion 1",
            "stable diffusion 1.4",
            "stable diffusion 1.5",
            "SD1",
            "SD14",
            "SD1.4",
            "SD15",
            "SD1.5",
            "stable_diffusion",
            "stable_diffusion_1",
            "stable_diffusion_1.4",
            "stable_diffusion_1.5",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_768,
    BaselineDescriptor(native_resolution=768),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_2_512,
    BaselineDescriptor(native_resolution=512),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "stable diffusion xl",
            "SDXL",
            "stable_diffusion_xl",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.stable_cascade,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "stable_cascade",
            "stable cascade",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.flux_1,
    BaselineDescriptor(native_resolution=1024),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.flux_schnell,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "flux_schnell",
            "flux schnell",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.flux_dev,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=(
            "flux_dev",
            "flux dev",
        ),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.qwen_image,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=("qwen_image", "qwen image", "qwen-image", "qwen"),
    ),
)
register_image_baseline(
    KNOWN_IMAGE_GENERATION_BASELINE.z_image_turbo,
    BaselineDescriptor(
        native_resolution=1024,
        alternative_names=("z_image_turbo", "z image turbo", "zimage-turbo", "zimage"),
    ),
)

_IMAGE_BASELINE_REGISTRY.finalize()

alternative_sdxl_baseline_names: list[str] = list(
    _IMAGE_BASELINE_REGISTRY.get(KNOWN_IMAGE_GENERATION_BASELINE.stable_diffusion_xl).alternative_names,
)


def _matching_image_baseline_exists(
    baseline: str,
    known_image_generation_baseline: KNOWN_IMAGE_GENERATION_BASELINE | str,
) -> bool:
    """Return True if *baseline* is a recognized alternative name for *known_image_generation_baseline*.

    Args:
        baseline: The baseline name to look up.
        known_image_generation_baseline: The known image generation baseline to check against.

    Returns:
        True if the baseline name matches the given known baseline, False otherwise.

    """
    desc = _IMAGE_BASELINE_REGISTRY.get(known_image_generation_baseline)
    if desc is not None and desc.alternative_names:
        return baseline in desc.alternative_names
    return baseline == str(known_image_generation_baseline)


def is_known_image_baseline(baseline: str) -> bool:
    """Return True if *baseline* is a known baseline or alternative name.

    Args:
        baseline: The baseline name to check.

    Returns:
        True if the baseline is known, False otherwise.

    """
    return _IMAGE_BASELINE_REGISTRY.contains(baseline) or baseline in _ALTERNATIVE_NAME_TO_BASELINE


def get_baseline_descriptor(baseline: KNOWN_IMAGE_GENERATION_BASELINE | str) -> BaselineDescriptor:
    """Return the ``BaselineDescriptor`` for *baseline*.

    Args:
        baseline: The known image generation baseline (enum member or plain string).

    Raises:
        KeyError: If the baseline is not registered.

    """
    return _IMAGE_BASELINE_REGISTRY.get(baseline)


def get_all_registered_baselines() -> dict[KNOWN_IMAGE_GENERATION_BASELINE | str, BaselineDescriptor]:
    """Return a shallow copy of the baseline registry.

    This includes both built-in ``KNOWN_IMAGE_GENERATION_BASELINE`` members and
    any externally registered baselines.
    """
    return _IMAGE_BASELINE_REGISTRY.all()


def get_baseline_native_resolution(baseline: KNOWN_IMAGE_GENERATION_BASELINE | str) -> int:
    """Get the native resolution of a stable diffusion baseline.

    Args:
        baseline: The stable diffusion baseline (enum member or plain string).

    Returns:
        The native resolution of the baseline.

    """
    return IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP[baseline]


def get_baselines_by_resolution(resolution: int) -> list[KNOWN_IMAGE_GENERATION_BASELINE | str]:
    """Get all baselines that have the given native resolution.

    Args:
        resolution: The native resolution to look for.

    Returns:
        A list of baselines that have the given native resolution.

    """
    return [
        baseline
        for baseline, native_resolution in IMAGE_GENERATION_BASELINE_NATIVE_RESOLUTION_LOOKUP.items()
        if native_resolution == resolution
    ]


_unregistered_baselines = {b for b in KNOWN_IMAGE_GENERATION_BASELINE if not _IMAGE_BASELINE_REGISTRY.contains(b)}
if _unregistered_baselines:
    raise RuntimeError(
        f"KNOWN_IMAGE_GENERATION_BASELINE members not registered in _BASELINE_REGISTRY: {_unregistered_baselines}"
    )


class CONTROLNET_STYLE(StrEnum):
    """An enum of all the ControlNet 'styles' - the process that defines the model's behavior.

    Examples include canny, depth, and openpose.
    """

    control_seg = auto()
    control_scribble = auto()
    control_fakescribbles = auto()
    control_openpose = auto()
    control_normal = auto()
    control_mlsd = auto()
    control_hough = auto()
    control_hed = auto()
    control_canny = auto()
    control_depth = auto()
    control_qr = auto()
    control_qr_xl = auto()
    control_lineart = auto()
    control_lineart_anime = auto()
    control_normal_bae = auto()
    control_recolor = auto()
    control_tile = auto()


_CONTROLNET_STYLE_REGISTRY = EnumRegistry(item.value for item in CONTROLNET_STYLE)


def register_controlnet_style(style: CONTROLNET_STYLE | str) -> None:
    """Register a new ControlNet style."""
    _CONTROLNET_STYLE_REGISTRY.register(style)


def is_known_controlnet_style(style: CONTROLNET_STYLE | str) -> bool:
    """Check if a ControlNet style is known."""
    return _CONTROLNET_STYLE_REGISTRY.is_known(style)
