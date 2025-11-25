"""Pydantic models for AI Horde public API responses.

API Documentation: https://aihorde.net/api/
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, RootModel

HordeModelType = Literal["image", "text"]
HordeWorkerType = Literal["image", "text", "interrogation", "alchemy"]  # All worker types from API
HordeModelState = Literal["known", "custom", "all"]


class HordeModelStatus(BaseModel):
    """Model status from Horde API status endpoint."""

    performance: float = Field(description="Performance metric (varies by model type)")
    queued: int = Field(description="Number of queued requests (pixelsteps for image, tokens for text)")
    jobs: int = Field(description="Number of active jobs")
    eta: int = Field(description="Estimated time to completion in seconds")
    type: HordeModelType = Field(description="Model type (image or text)")
    name: str = Field(description="Model name")
    count: int = Field(description="Number of workers serving this model")


class HordeModelStatsResponse(BaseModel):
    """Model statistics from Horde API stats endpoint."""

    day: dict[str, int] = Field(default_factory=dict, description="Statistics for the past day (model_name -> count)")
    month: dict[str, int] = Field(
        default_factory=dict, description="Statistics for the past month (model_name -> count)"
    )
    total: dict[str, int] = Field(default_factory=dict, description="All-time statistics (model_name -> count)")


class HordeTotalStatsTimePeriod(BaseModel):
    """Statistics for a specific time period."""

    images: int | None = Field(default=None, description="Number of images generated (for image type)")
    requests: int | None = Field(default=None, description="Number of requests processed")
    tokens: int | None = Field(default=None, description="Number of tokens processed (for text type)")
    ps: int | None = Field(default=None, description="Pixelsteps (for image type)")


class HordeTotalStatsResponse(BaseModel):
    """Total statistics across all models from Horde API."""

    minute: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past minute"
    )
    hour: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past hour"
    )
    day: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past day"
    )
    month: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="Statistics for the past month"
    )
    total: HordeTotalStatsTimePeriod = Field(
        default_factory=HordeTotalStatsTimePeriod, description="All-time statistics"
    )


class HordeModelUsageStats(BaseModel):
    """Usage statistics for a specific model."""

    minute: int | None = Field(default=None, description="Usage count for the past minute")
    hour: int | None = Field(default=None, description="Usage count for the past hour")
    day: int = Field(description="Usage count for the past day")
    month: int = Field(description="Usage count for the past month")
    total: int = Field(description="All-time usage count")


class BackendVariation(BaseModel):
    """Per-backend statistics for a text generation model variant.

    This tracks statistics for a specific backend (e.g., 'aphrodite' or 'koboldcpp')
    serving a particular model. Used to show backend-specific details in the UI
    while still providing aggregated totals at the model level.
    """

    backend: str = Field(description="Backend name (e.g., 'aphrodite', 'koboldcpp', or 'canonical' for non-prefixed)")
    variant_name: str = Field(description="Full model name as reported by Horde API (may include backend prefix)")
    worker_count: int = Field(description="Number of workers serving this backend variant")
    performance: float | None = Field(default=None, description="Performance metric for this variant")
    queued: int | None = Field(default=None, description="Number of queued requests for this variant")
    queued_jobs: int | None = Field(default=None, description="Number of active jobs for this variant")
    eta: int | None = Field(default=None, description="Estimated time to completion for this variant")
    usage_day: int = Field(default=0, description="Usage count for the past day")
    usage_month: int = Field(default=0, description="Usage count for the past month")
    usage_total: int = Field(default=0, description="All-time usage count")


class HordeWorkerTeam(BaseModel):
    """Worker team information."""

    name: str | None = Field(default=None, description="Team name")
    id: str | None = Field(default=None, description="Team ID (UUID)")


class HordeKudosDetails(BaseModel):
    """Kudos breakdown by source."""

    generated: float | None = Field(default=None, description="Kudos generated from work completed")
    uptime: float | None = Field(default=None, description="Kudos from uptime")


class IndexedHordeModelStatus(RootModel[dict[str, HordeModelStatus]]):
    """Indexed model status for O(1) lookups by model name.

    This wraps the status list and provides case-insensitive dictionary access.
    Time complexity: O(1) for lookups instead of O(n) for list iteration.

    Usage:
        indexed = IndexedHordeModelStatus([status1, status2, ...])
        status = indexed.get("model_name")  # Case-insensitive lookup
        all_statuses = indexed.get_all()    # Get all as list
    """

    root: dict[str, HordeModelStatus]

    def __init__(self, status_list: list[HordeModelStatus]) -> None:
        """Build indexed lookup from status list.

        Args:
            status_list: List of HordeModelStatus from API
        """
        # Build case-insensitive lookup dictionary
        status_dict = {s.name.lower(): s for s in status_list}
        super().__init__(root=status_dict)

    def get(self, model_name: str) -> HordeModelStatus | None:
        """Get status for a model by name (case-insensitive).

        Time Complexity: O(1)

        Args:
            model_name: Model name to look up

        Returns:
            HordeModelStatus if found, None otherwise
        """
        return self.root.get(model_name.lower())

    def get_all(self) -> list[HordeModelStatus]:
        """Get all status entries as a list.

        Returns:
            List of all HordeModelStatus objects
        """
        return list(self.root.values())

    def get_aggregated_status(self, canonical_name: str) -> HordeModelStatus | None:
        """Get aggregated status across all backend variants of a model.

        This method aggregates status from all possible backend-prefixed variants,
        taking the maximum worker count (count field) and the first non-None value
        for other fields.

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Aggregated HordeModelStatus or None if no variants have status.
        """
        from horde_model_reference.meta_consts import get_model_name_variants

        variants = get_model_name_variants(canonical_name)

        statuses: list[HordeModelStatus] = []
        for variant in variants:
            status = self.get(variant)
            if status is not None:
                statuses.append(status)

        if not statuses:
            return None

        return max(statuses, key=lambda s: s.count)

    def get_status_with_variations(
        self, canonical_name: str
    ) -> tuple[HordeModelStatus | None, dict[str, HordeModelStatus]]:
        """Get aggregated status and individual backend variations.

        This method returns both the aggregated status (same as get_aggregated_status)
        and a dictionary of individual backend statuses keyed by backend name.

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Tuple of (aggregated_status, variations_dict) where:
            - aggregated_status: Combined status or None if no variants found
            - variations_dict: Dict of backend_name -> HordeModelStatus
              Keys are 'canonical', 'aphrodite', 'koboldcpp' depending on what's found
        """
        from horde_model_reference.meta_consts import get_model_name_variants

        variants = get_model_name_variants(canonical_name)
        variations: dict[str, HordeModelStatus] = {}

        # Look up each variant and store by backend name
        for variant in variants:
            status = self.get(variant)
            if status is not None:
                # Determine backend name from variant
                if variant == canonical_name:
                    backend_name = "canonical"
                elif variant.startswith("aphrodite/"):
                    backend_name = "aphrodite"
                elif variant.startswith("koboldcpp/"):
                    backend_name = "koboldcpp"
                else:
                    backend_name = "unknown"

                variations[backend_name] = status

        if not variations:
            return None, {}

        # Aggregate: take max by worker count
        aggregated = max(variations.values(), key=lambda s: s.count)
        return aggregated, variations


class _StatsLookup(BaseModel):
    """Internal structure for indexed stats lookups."""

    day: dict[str, int] = Field(default_factory=dict)
    month: dict[str, int] = Field(default_factory=dict)
    total: dict[str, int] = Field(default_factory=dict)


def _strip_quantization_suffix(model_name: str) -> str:
    """Strip quantization suffix from a model name, preserving size.

    This is different from get_base_model_name which strips BOTH size and quantization.
    This function only strips quantization, keeping the size suffix.

    Args:
        model_name: Model name potentially with quantization suffix.

    Returns:
        Model name without quantization suffix, but with size preserved.

    Example:
        "Lumimaid-v0.2-8B-Q8_0" -> "Lumimaid-v0.2-8B"
        "Lumimaid-v0.2-8B" -> "Lumimaid-v0.2-8B"
        "koboldcpp/Lumimaid-v0.2-8B-Q4_K_M" -> "koboldcpp/Lumimaid-v0.2-8B"
    """
    import re

    # Quantization patterns to strip (same as text_model_parser.QUANT_PATTERNS but as suffix)
    quant_suffix_patterns = [
        r"[-_](Q[2-8]_K(?:_[SMLH])?)$",  # -Q4_K_M, -Q5_K_S
        r"[-_](Q[2-8]_[01])$",  # -Q4_0, -Q5_0, -Q8_0
        r"[-_](Q[2-8])$",  # -Q4, -Q8
        r"[-_](GGUF|GGML|GPTQ|AWQ|EXL2)$",
        r"[-_](fp16|fp32|int8|int4)$",
    ]

    result = model_name
    for pattern in quant_suffix_patterns:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    return result


def _build_base_name_index(model_names: list[str]) -> dict[str, list[str]]:
    """Build an index mapping base model names to all matching model names.

    This enables aggregating stats across quantization variants (e.g., Q4_K_M, Q8_0)
    and different backend prefixes (aphrodite/, koboldcpp/).

    Args:
        model_names: List of model names from API stats (may include backend prefixes
            and quantization suffixes).

    Returns:
        Dictionary mapping lowercase base model names to lists of original model names
        (lowercase) that match that base.

    Example:
        Input: ["koboldcpp/Lumimaid-v0.2-8B", "koboldcpp/Lumimaid-v0.2-8B-Q8_0",
                "aphrodite/NeverSleep/Lumimaid-v0.2-8B"]
        Output: {"lumimaid-v0.2": ["koboldcpp/lumimaid-v0.2-8b",
                                   "koboldcpp/lumimaid-v0.2-8b-q8_0",
                                   "aphrodite/neversleep/lumimaid-v0.2-8b"]}
    """
    from horde_model_reference.analytics.text_model_parser import get_base_model_name
    from horde_model_reference.meta_consts import strip_backend_prefix

    base_name_index: dict[str, list[str]] = {}

    for model_name in model_names:
        # Store lowercase version for case-insensitive matching
        model_name_lower = model_name.lower()

        # Strip backend prefix first, then extract base name
        stripped = strip_backend_prefix(model_name)

        # Also strip org prefix for base name extraction (e.g., "NeverSleep/Lumimaid-v0.2-8B" -> "Lumimaid-v0.2-8B")
        if "/" in stripped:
            stripped = stripped.split("/")[-1]

        base_name = get_base_model_name(stripped).lower()

        if base_name not in base_name_index:
            base_name_index[base_name] = []
        if model_name_lower not in base_name_index[base_name]:
            base_name_index[base_name].append(model_name_lower)

    return base_name_index


def _build_model_with_size_index(model_names: list[str]) -> dict[str, list[str]]:
    """Build an index mapping model names (with size, without quant) to all matching names.

    This enables aggregating stats across quantization variants only (e.g., Q4_K_M, Q8_0)
    while keeping different sizes separate.

    Unlike _build_base_name_index which groups ALL variants (including different sizes),
    this index only groups quantization variants of the SAME sized model.

    The key normalizes:
    - Backend prefix (stripped for matching, but preserved in values)
    - Org prefix (stripped for matching)
    - Quantization suffix (stripped for matching)

    But preserves:
    - Size suffix (8B, 12B, etc.)

    Args:
        model_names: List of model names from API stats (may include backend prefixes
            and quantization suffixes).

    Returns:
        Dictionary mapping normalized model names (backend/model-size) to lists of
        original model names (lowercase) that match that model.

    Example:
        Input: ["koboldcpp/Lumimaid-v0.2-8B", "koboldcpp/Lumimaid-v0.2-8B-Q8_0",
                "koboldcpp/Lumimaid-v0.2-12B", "aphrodite/NeverSleep/Lumimaid-v0.2-8B"]
        Output: {
            "koboldcpp/lumimaid-v0.2-8b": [
                "koboldcpp/lumimaid-v0.2-8b",
                "koboldcpp/lumimaid-v0.2-8b-q8_0"
            ],
            "koboldcpp/lumimaid-v0.2-12b": ["koboldcpp/lumimaid-v0.2-12b"],
            "aphrodite/lumimaid-v0.2-8b": ["aphrodite/neversleep/lumimaid-v0.2-8b"]
        }
    """
    model_with_size_index: dict[str, list[str]] = {}

    for model_name in model_names:
        model_name_lower = model_name.lower()

        # Extract backend prefix if present
        backend_prefix = ""
        stripped = model_name_lower
        if stripped.startswith("aphrodite/"):
            backend_prefix = "aphrodite/"
            stripped = stripped[len("aphrodite/") :]
        elif stripped.startswith("koboldcpp/"):
            backend_prefix = "koboldcpp/"
            stripped = stripped[len("koboldcpp/") :]

        # Strip org prefix (e.g., "neversleep/lumimaid-v0.2-8b" -> "lumimaid-v0.2-8b")
        if "/" in stripped:
            stripped = stripped.split("/")[-1]

        # Strip quantization suffix
        stripped_no_quant = _strip_quantization_suffix(stripped)

        # Build key: backend_prefix + model_name (no org, no quant, but with size)
        key = f"{backend_prefix}{stripped_no_quant}"

        if key not in model_with_size_index:
            model_with_size_index[key] = []
        if model_name_lower not in model_with_size_index[key]:
            model_with_size_index[key].append(model_name_lower)

    return model_with_size_index


class IndexedHordeModelStats(RootModel[_StatsLookup]):
    """Indexed model stats for O(1) lookups by model name.

    This wraps the stats response and provides case-insensitive dictionary access.
    Time complexity: O(1) for lookups instead of O(n) for dict iteration.

    Two indexes are built:
    - _base_name_index: Groups ALL variants (including different sizes) for group-level aggregation
    - _model_with_size_index: Groups only quantization variants for per-model stats

    Usage:
        indexed = IndexedHordeModelStats(stats_response)
        day_count = indexed.get_day("model_name")      # Case-insensitive
        has_data = indexed.has_stats("model_name")     # Check existence
    """

    root: _StatsLookup
    _base_name_index: dict[str, list[str]] = {}
    _model_with_size_index: dict[str, list[str]] = {}

    def __init__(self, stats_response: HordeModelStatsResponse) -> None:
        """Build indexed lookups from stats response.

        Args:
            stats_response: HordeModelStatsResponse from API
        """
        # Build case-insensitive lookup dictionaries for each time period
        lookups = _StatsLookup(
            day={k.lower(): v for k, v in stats_response.day.items()},
            month={k.lower(): v for k, v in stats_response.month.items()},
            total={k.lower(): v for k, v in stats_response.total.items()},
        )
        super().__init__(root=lookups)

        # Build indexes from all unique model names across all time periods
        all_model_names = (
            set(stats_response.day.keys()) | set(stats_response.month.keys()) | set(stats_response.total.keys())
        )
        model_names_list = list(all_model_names)
        self._base_name_index = _build_base_name_index(model_names_list)
        self._model_with_size_index = _build_model_with_size_index(model_names_list)

    def get_day(self, model_name: str) -> int | None:
        """Get day count for a model (case-insensitive). O(1)."""
        return self.root.day.get(model_name.lower())

    def get_month(self, model_name: str) -> int | None:
        """Get month count for a model (case-insensitive). O(1)."""
        return self.root.month.get(model_name.lower())

    def get_total(self, model_name: str) -> int | None:
        """Get total count for a model (case-insensitive). O(1)."""
        return self.root.total.get(model_name.lower())

    def has_stats(self, model_name: str) -> bool:
        """Check if model has any stats (case-insensitive). O(1)."""
        name_lower = model_name.lower()
        return name_lower in self.root.day or name_lower in self.root.month or name_lower in self.root.total

    def get_aggregated_stats(self, canonical_name: str) -> tuple[int, int, int]:
        """Get aggregated stats across all backend variants and quantization versions of a model.

        This method aggregates stats from:
        - Exact name variants (canonical, aphrodite/, koboldcpp/ prefixed)
        - All quantization variants sharing the same base model name (Q4_K_M, Q8_0, etc.)
        - Different org-prefixed variants (e.g., "NeverSleep/Lumimaid" matches "Lumimaid")

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Tuple of (day_total, month_total, total_total) aggregated across all variants.

        Example:
            >>> indexed = IndexedHordeModelStats(stats_response)
            >>> day, month, total = indexed.get_aggregated_stats("Lumimaid-v0.2-8B")
            # Will aggregate: Lumimaid-v0.2-8B, koboldcpp/Lumimaid-v0.2-8B,
            #                 koboldcpp/Lumimaid-v0.2-8B-Q8_0, aphrodite/NeverSleep/Lumimaid-v0.2-8B, etc.
        """
        from horde_model_reference.analytics.text_model_parser import get_base_model_name
        from horde_model_reference.meta_consts import get_model_name_variants

        # Collect all model names to aggregate (use set to avoid double-counting)
        names_to_aggregate: set[str] = set()

        # First, add exact variants from get_model_name_variants
        variants = get_model_name_variants(canonical_name)
        for variant in variants:
            names_to_aggregate.add(variant.lower())

        # Then, add all model names that share the same base model name
        # This catches quantization variants and org-prefixed variants
        # Strip org prefix from canonical name if present (e.g., "NeverSleep/Lumimaid-v0.2" -> "Lumimaid-v0.2")
        canonical_without_org = canonical_name.split("/")[-1] if "/" in canonical_name else canonical_name
        base_name = get_base_model_name(canonical_without_org).lower()
        if base_name in self._base_name_index:
            for api_model_name in self._base_name_index[base_name]:
                names_to_aggregate.add(api_model_name)

        # Aggregate stats across all matched names
        day_total = 0
        month_total = 0
        total_total = 0

        for name in names_to_aggregate:
            day_total += self.get_day(name) or 0
            month_total += self.get_month(name) or 0
            total_total += self.get_total(name) or 0

        return (day_total, month_total, total_total)

    def get_stats_with_variations(
        self, canonical_name: str
    ) -> tuple[tuple[int, int, int], dict[str, tuple[int, int, int]]]:
        """Get stats for a specific model broken down by backend.

        Unlike get_aggregated_stats which aggregates across all models with the same
        base name (e.g., all Lumimaid-v0.2 sizes), this method returns stats only for
        the exact model specified (including its quantization variants), broken down
        by backend prefix.

        This enables showing per-model stats in the UI when displaying grouped models,
        where each model variant (8B, 12B, etc.) shows its own stats by backend.

        Args:
            canonical_name: The canonical model name from the model reference.

        Returns:
            Tuple of (aggregated_stats, variations_dict) where:
            - aggregated_stats: (day_total, month_total, total_total) for this exact model
            - variations_dict: Dict of backend_name -> (day, month, total)
              Keys are 'canonical', 'aphrodite', 'koboldcpp' depending on what's found
        """
        from horde_model_reference.meta_consts import get_model_name_variants

        # Collect all model names that are variants of this specific model
        # Use _model_with_size_index to include quantization variants, but NOT size variants
        names_to_aggregate: set[str] = set()

        # Get exact backend-prefixed variants
        variants = get_model_name_variants(canonical_name)
        for variant in variants:
            variant_lower = variant.lower()
            names_to_aggregate.add(variant_lower)

            # Build the normalized key to look up in _model_with_size_index
            # The key format is: [backend_prefix/]model_name (no org, no quant)
            backend_prefix = ""
            stripped = variant_lower
            if stripped.startswith("aphrodite/"):
                backend_prefix = "aphrodite/"
                stripped = stripped[len("aphrodite/") :]
            elif stripped.startswith("koboldcpp/"):
                backend_prefix = "koboldcpp/"
                stripped = stripped[len("koboldcpp/") :]

            # Strip org prefix if present
            if "/" in stripped:
                stripped = stripped.split("/")[-1]

            # Strip quantization suffix and build key
            stripped_no_quant = _strip_quantization_suffix(stripped)
            key = f"{backend_prefix}{stripped_no_quant}"

            if key in self._model_with_size_index:
                for api_model_name in self._model_with_size_index[key]:
                    names_to_aggregate.add(api_model_name)

        # Track stats by backend for variations dict
        backend_stats: dict[str, tuple[int, int, int]] = {
            "canonical": (0, 0, 0),
            "aphrodite": (0, 0, 0),
            "koboldcpp": (0, 0, 0),
        }

        day_total = 0
        month_total = 0
        total_total = 0

        # Look up each name and aggregate by backend
        for name in names_to_aggregate:
            day = self.get_day(name) or 0
            month = self.get_month(name) or 0
            total = self.get_total(name) or 0

            if day > 0 or month > 0 or total > 0:
                # Determine backend name from the model name
                if name.startswith("aphrodite/"):
                    backend_name = "aphrodite"
                elif name.startswith("koboldcpp/"):
                    backend_name = "koboldcpp"
                else:
                    backend_name = "canonical"

                # Accumulate stats for this backend
                prev_day, prev_month, prev_total = backend_stats[backend_name]
                backend_stats[backend_name] = (prev_day + day, prev_month + month, prev_total + total)

            day_total += day
            month_total += month
            total_total += total

        # Filter to only backends with data
        variations = {k: v for k, v in backend_stats.items() if v != (0, 0, 0)}

        return (day_total, month_total, total_total), variations


class HordeWorker(BaseModel):
    """Worker information from Horde API.

    This is a simplified model that captures the common fields needed for merging.
    The API returns additional type-specific fields that aren't needed for basic integration.
    """

    name: str = Field(description="Worker name")
    id: str = Field(description="Worker ID (UUID)")
    type: HordeWorkerType = Field(description="Worker type (image, text, interrogation, or alchemy)")
    performance: str = Field(description="Performance metric as a string")
    requests_fulfilled: int = Field(description="Number of requests fulfilled by this worker")
    kudos_rewards: float = Field(description="Total kudos rewards earned")
    kudos_details: HordeKudosDetails = Field(description="Kudos breakdown by source")
    threads: int = Field(description="Number of worker threads")
    uptime: int = Field(description="Total uptime in seconds")
    uncompleted_jobs: int = Field(description="Number of jobs not yet completed")
    maintenance_mode: bool = Field(description="Whether worker is in maintenance mode")
    nsfw: bool = Field(description="Whether worker serves NSFW content")
    trusted: bool = Field(description="Whether worker is trusted")
    flagged: bool = Field(description="Whether worker is flagged")
    online: bool = Field(description="Whether worker is currently online")
    models: list[str] = Field(default_factory=list, description="List of model names this worker serves")
    team: HordeWorkerTeam = Field(description="Team information (if worker belongs to a team)")
    bridge_agent: str = Field(description="Bridge agent information (software/version)")

    # Type-specific optional fields (not all workers have these)
    max_pixels: int | None = Field(default=None, description="Maximum pixels (image workers)")
    megapixelsteps_generated: float | None = Field(default=None, description="Total megapixelsteps (image workers)")
    img2img: bool | None = Field(default=None, description="Supports img2img (image workers)")
    painting: bool | None = Field(default=None, description="Supports inpainting (image workers)")
    lora: bool | None = Field(default=None, description="Supports LoRA (image workers)")
    controlnet: bool | None = Field(default=None, description="Supports ControlNet (image workers)")
    sdxl_controlnet: bool | None = Field(default=None, description="Supports SDXL ControlNet (image workers)")
    max_length: int | None = Field(default=None, description="Maximum token length (text workers)")
    max_context_length: int | None = Field(default=None, description="Maximum context length (text workers)")
    info: str | None = Field(default=None, description="Additional worker information")

    # Handle post-processing as aliased field (hyphen in API response)
    post_processing: bool | None = Field(default=None, alias="post-processing", description="Supports post-processing")


class IndexedHordeWorkers(RootModel[dict[str, list[HordeWorker]]]):
    """Indexed workers for O(1) lookups by model name.

    This wraps the workers list and provides case-insensitive dictionary access
    where keys are model names and values are lists of workers serving that model.
    Time complexity: O(1) for lookups instead of O(w*m) iteration.

    Usage:
        indexed = IndexedHordeWorkers([worker1, worker2, ...])
        workers = indexed.get("model_name")  # Case-insensitive lookup
        all_workers = indexed.get_all()      # Get all unique workers
    """

    root: dict[str, list[HordeWorker]]

    def __init__(self, workers_list: list[HordeWorker]) -> None:
        """Build indexed lookup from workers list.

        Args:
            workers_list: List of HordeWorker from API
        """
        # Build case-insensitive lookup dictionary by model name
        workers_by_model: dict[str, list[HordeWorker]] = {}
        for worker in workers_list:
            for model_name in worker.models:
                model_name_lower = model_name.lower()
                if model_name_lower not in workers_by_model:
                    workers_by_model[model_name_lower] = []
                workers_by_model[model_name_lower].append(worker)
        super().__init__(root=workers_by_model)

    def get(self, model_name: str) -> list[HordeWorker]:
        """Get workers for a model by name (case-insensitive).

        Time Complexity: O(1)

        Args:
            model_name: Model name to look up

        Returns:
            List of HordeWorker serving this model (empty list if none)
        """
        return self.root.get(model_name.lower(), [])

    def get_all(self) -> list[HordeWorker]:
        """Get all unique workers as a list.

        Returns:
            List of all HordeWorker objects (deduplicated)
        """
        seen_ids = set()
        all_workers = []
        for workers in self.root.values():
            for worker in workers:
                if worker.id not in seen_ids:
                    seen_ids.add(worker.id)
                    all_workers.append(worker)
        return all_workers
