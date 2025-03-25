#!/usr/bin/python

from nilai_api.db.logs import QueryLogManager
from nilai_api.db.users import UserManager


# Example Usage
async def main():
    # Add some users
    bob = await UserManager.insert_user("Bob", "bob@example.com")
    alice = await UserManager.insert_user("Alice", "alice@example.com")

    print(f"Bob's details: {bob}")
    print(f"Alice's details: {alice}")

    # Check API key
    user_name = await UserManager.check_api_key(bob.apikey)
    print(f"API key validation: {user_name}")

    # Update and retrieve token usage
    await UserManager.update_token_usage(
        bob.userid, prompt_tokens=50, completion_tokens=20
    )
    usage = await UserManager.get_user_token_usage(bob.userid)
    print(f"Bob's token usage: {usage}")

    # Log a query
    await QueryLogManager.log_query(
        userid=bob.userid,
        model="gpt-3.5-turbo",
        prompt_tokens=8,
        completion_tokens=7,
    )


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(main())
