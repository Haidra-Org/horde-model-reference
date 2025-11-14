"""Unit tests for the text model parser module."""

from __future__ import annotations

from horde_model_reference.analytics.text_model_parser import (
    get_base_model_name,
    get_model_size,
    get_model_variant,
    group_text_models_by_base,
    is_quantized_variant,
    normalize_model_name,
    parse_text_model_name,
)


class TestParseTextModelName:
    """Tests for text model name parsing."""

    def test_parse_simple_name(self) -> None:
        """Test parsing a simple model name."""
        parsed = parse_text_model_name("Llama-3")

        assert parsed.original_name == "Llama-3"
        assert parsed.base_name == "Llama-3"
        assert parsed.size is None
        assert parsed.variant is None
        assert parsed.quant is None

    def test_parse_name_with_size(self) -> None:
        """Test parsing a model name with size."""
        parsed = parse_text_model_name("Llama-3-8B")

        assert parsed.original_name == "Llama-3-8B"
        assert parsed.base_name == "Llama-3"
        assert parsed.size == "8B"
        assert parsed.variant is None
        assert parsed.quant is None

    def test_parse_name_with_variant(self) -> None:
        """Test parsing a model name with variant."""
        parsed = parse_text_model_name("Llama-3-8B-Instruct")

        assert parsed.original_name == "Llama-3-8B-Instruct"
        assert parsed.base_name == "Llama-3"
        assert parsed.size == "8B"
        assert parsed.variant == "Instruct"
        assert parsed.quant is None

    def test_parse_name_with_quant(self) -> None:
        """Test parsing a model name with quantization."""
        parsed = parse_text_model_name("Llama-3-8B-Instruct-Q4_K_M")

        assert parsed.original_name == "Llama-3-8B-Instruct-Q4_K_M"
        assert parsed.base_name == "Llama-3"
        assert parsed.size == "8B"
        assert parsed.variant == "Instruct"
        assert parsed.quant == "Q4_K_M"

    def test_parse_mistral_model(self) -> None:
        """Test parsing Mistral model names."""
        parsed = parse_text_model_name("Mistral-7B-v0.1")

        assert parsed.base_name == "Mistral-v0.1"
        assert parsed.size == "7B"

    def test_parse_mixtral_moe(self) -> None:
        """Test parsing Mixtral MoE model names."""
        parsed = parse_text_model_name("Mixtral-8x7B-Instruct-v0.1")

        assert parsed.base_name == "Mixtral--v0.1"
        assert parsed.size == "8X7B"
        assert parsed.variant == "Instruct"

    def test_parse_gemma_model(self) -> None:
        """Test parsing Gemma model names."""
        parsed = parse_text_model_name("Gemma-2B-Instruct")

        assert parsed.base_name == "Gemma"
        assert parsed.size == "2B"
        assert parsed.variant == "Instruct"

    def test_parse_phi_model(self) -> None:
        """Test parsing Phi model names."""
        parsed = parse_text_model_name("Phi-3-Mini-4K-Instruct")

        assert parsed.base_name == "Phi-3-Mini"
        assert parsed.size == "4K"
        assert parsed.variant == "Instruct"

    def test_parse_code_variant(self) -> None:
        """Test parsing code variant models."""
        parsed = parse_text_model_name("CodeLlama-7B-Instruct")

        assert parsed.base_name == "CodeLlama"
        assert parsed.size == "7B"
        assert parsed.variant == "Instruct"

    def test_parse_chat_variant(self) -> None:
        """Test parsing chat variant models."""
        parsed = parse_text_model_name("Vicuna-13B-Chat")

        assert parsed.base_name == "Vicuna"
        assert parsed.size == "13B"
        assert parsed.variant == "Chat"

    def test_parse_gguf_quant(self) -> None:
        """Test parsing GGUF quantized models."""
        parsed = parse_text_model_name("Llama-3-8B-Instruct-GGUF")

        assert parsed.base_name == "Llama-3"
        assert parsed.size == "8B"
        assert parsed.variant == "Instruct"
        assert parsed.quant == "GGUF"

    def test_parse_gptq_quant(self) -> None:
        """Test parsing GPTQ quantized models."""
        parsed = parse_text_model_name("Llama-2-13B-GPTQ")

        assert parsed.base_name == "Llama-2"
        assert parsed.size == "13B"
        assert parsed.quant == "GPTQ"

    def test_parse_awq_quant(self) -> None:
        """Test parsing AWQ quantized models."""
        parsed = parse_text_model_name("Mistral-7B-Instruct-AWQ")

        assert parsed.base_name == "Mistral"
        assert parsed.size == "7B"
        assert parsed.variant == "Instruct"
        assert parsed.quant == "AWQ"

    def test_parse_fp16_precision(self) -> None:
        """Test parsing models with precision indicators."""
        parsed = parse_text_model_name("Llama-3-8B-fp16")

        assert parsed.base_name == "Llama-3"
        assert parsed.size == "8B"
        assert parsed.quant == "FP16"

    def test_parse_uncensored_variant(self) -> None:
        """Test parsing uncensored variant models."""
        parsed = parse_text_model_name("Llama-2-7B-Uncensored")

        assert parsed.base_name == "Llama-2"
        assert parsed.size == "7B"
        assert parsed.variant == "Uncensored"

    def test_parse_decimal_size(self) -> None:
        """Test parsing models with decimal sizes."""
        parsed = parse_text_model_name("Qwen-1.5B-Chat")

        assert parsed.base_name == "Qwen"
        assert parsed.size == "1.5B"
        assert parsed.variant == "Chat"


class TestGetBaseModelName:
    """Tests for get_base_model_name function."""

    def test_get_base_from_full_name(self) -> None:
        """Test extracting base name from full model name."""
        assert get_base_model_name("Llama-3-8B-Instruct-Q4_K_M") == "Llama-3"
        assert get_base_model_name("Mistral-7B-v0.1") == "Mistral-v0.1"
        assert get_base_model_name("Gemma-2B-Instruct") == "Gemma"

    def test_get_base_from_simple_name(self) -> None:
        """Test extracting base name from simple model name."""
        assert get_base_model_name("Llama-3") == "Llama-3"
        assert get_base_model_name("Mistral") == "Mistral"


class TestNormalizeModelName:
    """Tests for normalize_model_name function."""

    def test_normalize_converts_lowercase(self) -> None:
        """Test normalization converts to lowercase."""
        assert normalize_model_name("Llama-3-8B-Instruct") == "llama_3_8b_instruct"

    def test_normalize_replaces_separators(self) -> None:
        """Test normalization replaces different separators with underscores."""
        assert normalize_model_name("Llama-3 8B.Instruct") == "llama_3_8b_instruct"

    def test_normalize_removes_multiple_underscores(self) -> None:
        """Test normalization removes consecutive underscores."""
        assert normalize_model_name("Llama--3__8B") == "llama_3_8b"

    def test_normalize_strips_underscores(self) -> None:
        """Test normalization strips leading/trailing underscores."""
        assert normalize_model_name("-Llama-3-") == "llama_3"


class TestGroupTextModelsByBase:
    """Tests for group_text_models_by_base function."""

    def test_group_empty_list(self) -> None:
        """Test grouping an empty list."""
        grouped = group_text_models_by_base([])

        assert grouped == {}

    def test_group_single_model(self) -> None:
        """Test grouping a single model."""
        models = ["Llama-3-8B-Instruct"]
        grouped = group_text_models_by_base(models)

        assert len(grouped) == 1
        assert "Llama-3" in grouped
        assert grouped["Llama-3"].base_name == "Llama-3"
        assert grouped["Llama-3"].variants == ["Llama-3-8B-Instruct"]

    def test_group_variants_together(self) -> None:
        """Test that variants of the same model are grouped together."""
        models = [
            "Llama-3-8B-Instruct",
            "Llama-3-8B-Instruct-Q4_K_M",
            "Llama-3-8B-Instruct-Q8",
            "Llama-3-70B-Instruct",
        ]
        grouped = group_text_models_by_base(models)

        assert len(grouped) == 1
        assert "Llama-3" in grouped
        assert len(grouped["Llama-3"].variants) == 4

    def test_group_different_models_separately(self) -> None:
        """Test that different models are grouped separately."""
        models = [
            "Llama-3-8B-Instruct",
            "Mistral-7B-v0.1",
            "Gemma-2B-Chat",
        ]
        grouped = group_text_models_by_base(models)

        assert len(grouped) == 3
        assert "Llama-3" in grouped
        assert "Mistral-v0.1" in grouped
        assert "Gemma" in grouped

    def test_group_mixed_models(self) -> None:
        """Test grouping a mix of models and variants."""
        models = [
            "Llama-3-8B-Instruct",
            "Llama-3-8B-Instruct-Q4",
            "Llama-2-7B-Chat",
            "Mistral-7B-v0.1",
            "Mistral-7B-v0.1-GGUF",
        ]
        grouped = group_text_models_by_base(models)

        assert len(grouped) == 3
        assert len(grouped["Llama-3"].variants) == 2
        assert len(grouped["Llama-2"].variants) == 1
        assert len(grouped["Mistral-v0.1"].variants) == 2


class TestIsQuantizedVariant:
    """Tests for is_quantized_variant function."""

    def test_quantized_q4(self) -> None:
        """Test detection of Q4 quantized models."""
        assert is_quantized_variant("Llama-3-8B-Instruct-Q4_K_M")

    def test_quantized_gguf(self) -> None:
        """Test detection of GGUF quantized models."""
        assert is_quantized_variant("Mistral-7B-GGUF")

    def test_quantized_gptq(self) -> None:
        """Test detection of GPTQ quantized models."""
        assert is_quantized_variant("Llama-2-13B-GPTQ")

    def test_quantized_awq(self) -> None:
        """Test detection of AWQ quantized models."""
        assert is_quantized_variant("Mistral-7B-AWQ")

    def test_not_quantized(self) -> None:
        """Test detection of non-quantized models."""
        assert not is_quantized_variant("Llama-3-8B-Instruct")
        assert not is_quantized_variant("Mistral-7B-v0.1")


class TestGetModelSize:
    """Tests for get_model_size function."""

    def test_get_size_standard(self) -> None:
        """Test extracting standard model sizes."""
        assert get_model_size("Llama-3-8B-Instruct") == "8B"
        assert get_model_size("Mistral-7B-v0.1") == "7B"
        assert get_model_size("GPT-2-1.5B") == "1.5B"

    def test_get_size_moe(self) -> None:
        """Test extracting MoE model sizes."""
        assert get_model_size("Mixtral-8x7B-Instruct") == "8X7B"

    def test_get_size_none(self) -> None:
        """Test returns None when size not found."""
        assert get_model_size("GPT-4") is None
        assert get_model_size("Claude") is None


class TestGetModelVariant:
    """Tests for get_model_variant function."""

    def test_get_variant_instruct(self) -> None:
        """Test extracting Instruct variant."""
        assert get_model_variant("Llama-3-8B-Instruct") == "Instruct"

    def test_get_variant_chat(self) -> None:
        """Test extracting Chat variant."""
        assert get_model_variant("Vicuna-13B-Chat") == "Chat"

    def test_get_variant_code(self) -> None:
        """Test extracting Code variant."""
        assert get_model_variant("CodeLlama-7B-Code") == "Code"

    def test_get_variant_uncensored(self) -> None:
        """Test extracting Uncensored variant."""
        assert get_model_variant("Llama-2-7B-Uncensored") == "Uncensored"

    def test_get_variant_none(self) -> None:
        """Test returns None when variant not found."""
        assert get_model_variant("Llama-3-8B") is None
        assert get_model_variant("Mistral-7B") is None
