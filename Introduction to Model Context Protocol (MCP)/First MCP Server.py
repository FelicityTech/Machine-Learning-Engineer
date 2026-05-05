from mcp.server.fastmcp import FastMCP
import requests
# Create an MCP server instance
mcp = FastMCP("Currency Converter")

@mcp.tool()
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another using current exchange rates.

    Args:
        amount: The amount to convert
        from_currency: Source currency code (e.g., 'USD', 'EUR', 'GBP')
        to_currency: Target currency code (e.g., 'USD', 'EUR', 'GBP')

    Returns:
        A string with the conversion result and exchange rate
    """
    # API endpoint for Frankfurter
    url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"

    try:
        # Make the API request
        response = requests.get(url)
        response.raise_for_status()

        # Parse the response
        data = response.json()

        # Get the exchange rate
        rate = data['rates'].get(to_currency)

        if rate is None:
            return f"Could not find exchange rate for {from_currency} to {to_currency}"

        # Calculate the converted amount
        converted_amount = amount * rate

        return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"

    except requests.exceptions.RequestException as e:
        return f"Error converting currency: {str(e)}"

print("Testing Currency Converter:")
result = convert_currency(amount=100, from_currency="USD", to_currency="EUR")
print(result)