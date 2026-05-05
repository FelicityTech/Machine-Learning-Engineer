from mcp.server.fastmcp import FastMCP
import requests
# Create an MCP server instance
mcp = FastMCP("Currency Converter")

# Define the MCP tool
@mcp.tool()
def convert_currency(amount, from_currency, to_currency):
    # API endpoint for Frankfurter
    url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"

    # 1. Make the API request
    response = requests.get(url)

    # 2. Extract the currency exchange rate from the response
    data = response.json()
    rate = data['rates'].get(to_currency)

    if rate is None:
        return f"Could not find exchange rate for {from_currency} to {to_currency}"

    # 3. Calculate the converted amount
    converted_amount = amount * rate
    return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"
      
print(convert_currency(amount=100, from_currency="EUR", to_currency="USD"))