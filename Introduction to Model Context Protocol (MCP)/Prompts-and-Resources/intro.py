from mcp.server.fastmcp impost FastMCP

mcp = FastMCP("Timezone Converter")


@mcp.tool()
def convert_timezone(utc_time, timezone):
    """Convert a UTC time to a specific timezone.

    """


@mcp.resource()
def base_prompt_resource(base_prompt: str, currency_request: str) -> dict:
    """
    A resource that returns a prompt template filled with user input.

    Args:
        base_prompt: The base prompt template
        currency_request: User input related to currency

    Returns:
        A dictionary with the prompt content
    """

    # Fill the template with user input
    final_prompt = base_prompt.format(currency_request=currency_request)

    return {
        "text": final_prompt
    }


@mcp.prompt()
def get_prompt_resource(base_prompt: str, currency_request: str) -> dict:
    """
    A prompt that returns a prompt template filled with user input.

    Args:
        base_prompt: The base prompt template
        currency_request: User input related to currency

    Returns:
        A dictionary with the prompt content
    """

    # Fill the template with user input
    final_prompt = base_prompt.format(currency_request=currency_request)

    return {
        "text": final_prompt
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
    