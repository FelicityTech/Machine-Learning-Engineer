from mcp.server.fastmcp import FastMCP
import requests
# Create an MCP server instance
mcp = FastMCP("Currency Converter")

# Adding typing to the function arguments and return object
@mcp.tool()
def convert_currency(amount: float, from_currency:str, to_currency: str) -> str:
    # Complete the docstring with the function arguments
    """
    Convert an amount from one currency to another using current exchange rates.

    Args:
        amount: The amount to convert
        from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
        to_currency: Target currency code (e.g., 'USD', 'EUR', 'GBP')

    Returns:
        A string with the conversion result and exchange rate
    """

    url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"

    response = requests.get(url)
    data = response.json()
    rate = data['rates'].get(to_currency)

    if rate is None:
        return f"Could not find exchange rate for {from_currency} to {to_currency}"

    converted_amount = amount * rate
    return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"

print(convert_currency(amount=100, from_currency="EUR", to_currency="USD"))