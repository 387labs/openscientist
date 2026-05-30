"""
Provider abstraction for model access (Anthropic, CBORG, Vertex AI, Bedrock, Azure Foundry).

Providers handle:
- Environment configuration for Claude CLI
- Cost tracking and budget enforcement
- Provider-specific authentication and setup
"""

from openscientist.providers.base import CostInfo, Provider
from openscientist.settings import get_settings


def get_provider() -> Provider:
    """
    Get the configured provider based on environment.

    Returns:
        Provider instance (Anthropic/CBORG/Vertex/Bedrock/Foundry, all
        ClaudeCompatible, or OpenAI, which is CodexCompatible)

    Raises:
        ValueError: If provider is unknown or misconfigured

    Environment:
        OPENSCIENTIST_PROVIDER: Provider name ("anthropic", "cborg", "vertex",
                               "bedrock", "foundry", "openai"). Defaults to
                               "anthropic" if not set.
    """
    settings = get_settings()
    provider_name = settings.provider.provider_id.lower()

    if provider_name == "anthropic":
        from openscientist.providers.anthropic import AnthropicProvider

        return AnthropicProvider()
    if provider_name == "cborg":
        from openscientist.providers.cborg import CborgProvider

        return CborgProvider()
    if provider_name == "vertex":
        from openscientist.providers.vertex import VertexProvider

        return VertexProvider()
    if provider_name == "bedrock":
        from openscientist.providers.bedrock import BedrockProvider

        return BedrockProvider()
    if provider_name == "foundry":
        from openscientist.providers.foundry import FoundryProvider

        return FoundryProvider()
    if provider_name == "openai":
        from openscientist.providers.openai import OpenAIDirectProvider

        return OpenAIDirectProvider()
    raise ValueError(
        f"Unknown provider '{provider_name}'. Valid options: anthropic, cborg, "
        "vertex, bedrock, foundry, openai"
    )


def check_provider_config() -> tuple[bool, str, list[str]]:
    """
    Check if the provider is properly configured without raising exceptions.

    Returns:
        Tuple of (is_configured, provider_name, error_messages)
        - is_configured: True if provider can be instantiated
        - provider_name: Name of the configured provider
        - error_messages: List of configuration error messages (empty if configured)

    Environment:
        SIMULATE_PROVIDER_ERROR: Set to "true" for E2E testing of error UI
    """
    settings = get_settings()

    # Testing hook: simulate provider misconfiguration for E2E tests
    if settings.dev.simulate_provider_error:
        return (
            False,
            "anthropic",
            [
                "ANTHROPIC_API_KEY is missing or invalid",
                "Please contact your administrator to configure API credentials",
            ],
        )

    provider_name = settings.provider.provider_id.lower()

    valid_providers = ("anthropic", "cborg", "vertex", "bedrock", "foundry", "openai")
    if provider_name not in valid_providers:
        return (
            False,
            provider_name,
            [f"Unknown provider '{provider_name}'. Valid options: {', '.join(valid_providers)}"],
        )

    try:
        get_provider()
        return (True, provider_name, [])
    except ValueError as e:
        # Extract error messages from the exception
        error_str = str(e)
        errors = [line.strip() for line in error_str.split("\n") if line.strip()]
        return (False, provider_name, errors)


__all__ = ["ClaudeCompatible", "CostInfo", "check_provider_config", "get_provider"]
