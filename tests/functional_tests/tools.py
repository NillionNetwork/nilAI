from openai import OpenAI


def test_tools():
    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://test.nilai.sandbox.nilogy.xyz/v1/",
        api_key="abcdef12-3456-7890-abcd-ef1234567890",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "user", "content": "What is the weather like in Paris today?"}
        ],
        tools=tools, # type: ignore
    )
    print(response)
    # raise Exception(f"Response: {response}")


if __name__ == "__main__":
    test_tools()
