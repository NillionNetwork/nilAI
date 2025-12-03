from nilai_py import Client

from config import API_KEY


def main():
    # Initialize the client in API key mode
    # To obtain an API key, navigate to https://nilpay.vercel.app/
    # and create a new subscription.
    # The API key will be displayed in the subscription details.
    # The Client class automatically handles the NUC token creation and management.
    ## For sandbox, use the following:
    client = Client(
        base_url="https://nilai-a779.nillion.network/v1/",
        api_key=API_KEY,
        # For production, use the following:
        # nilauth_instance=NilAuthInstance.PRODUCTION,
    )

    # Make a streaming request to the Nilai API
    print("Starting streaming response...")
    print("=" * 50)

    stream = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {
                "role": "user",
                "content": "Write a short story about a robot learning to paint. Make it creative and engaging.",
            }
        ],
        stream=True,  # Enable streaming
    )

    # Process the streaming response
    full_response = ""
    for chunk in stream:
        if (
            chunk.choices is not None
            and len(chunk.choices) > 0
            and chunk.choices[0].delta.content is not None
        ):
            content = chunk.choices[0].delta.content
            print(
                content, end="", flush=True
            )  # Print without newline and flush immediately
            full_response += content

    print("\n" + "=" * 50)
    print(
        f"\nStreaming completed. Full response length: {len(full_response)} characters"
    )


if __name__ == "__main__":
    main()
