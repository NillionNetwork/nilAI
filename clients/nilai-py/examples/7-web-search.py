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
        base_url="https://nilai-f910.nillion.network/nuc/v1/",
        api_key=API_KEY,
    )

    # Make a request to the Nilai API
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Can you look for the latest news about AI and summarize them?",
            }
        ],
        extra_body={"web_search": True},
    )

    print(f"Response: {response.choices[0].message.content}")


if __name__ == "__main__":
    main()
