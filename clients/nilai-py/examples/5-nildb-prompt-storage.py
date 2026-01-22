## NOTE: DELEGATION TOKEN MODE DOES NOT WORK
##      AS THIS IS RESERVED TO SUBSCRIPTION OWNERS


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
        base_url="https://nilai-f910.nillion.network/nuc/v1/", api_key=API_KEY
    )

    # Make a request to the Nilai API
    # response = client.chat.completions.create(
    #     model="google/gemma-3-27b-it",
    #     messages=[
    #         {"role": "user", "content": "What is your name?"}
    #     ],
    # )

    # print(f"Response: {response.choices[0].message.content}")
    # List prompts from Nildb
    client.list_prompts_from_nildb()

    store_ids = client.store_prompt_to_nildb(
        prompt="You are a very clever model that answers with cheese answers and always starting with the word cheese"
    )
    print("Stored document IDs:", store_ids)

    client.list_prompts_from_nildb()


if __name__ == "__main__":
    main()
