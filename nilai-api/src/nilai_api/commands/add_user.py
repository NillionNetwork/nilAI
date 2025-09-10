import asyncio
import json

from nilai_api.db.users import RateLimits, UserManager
import click


@click.command()
@click.option("--name", type=str, required=True, help="User Name")
@click.option("--apikey", type=str, help="API Key")
@click.option("--userid", type=str, help="User Id")
@click.option("--ratelimit-day", type=int, help="number of request per day")
@click.option("--ratelimit-hour", type=int, help="number of request per hour")
@click.option("--ratelimit-minute", type=int, help="number of request per minute")
@click.option(
    "--web-search-ratelimit-day", type=int, help="number of web search request per day"
)
@click.option(
    "--web-search-ratelimit-hour",
    type=int,
    help="number of web search request per hour",
)
@click.option(
    "--web-search-ratelimit-minute",
    type=int,
    help="number of web search request per minute",
)
def main(
    name,
    apikey: str | None,
    userid: str | None,
    ratelimit_day: int | None,
    ratelimit_hour: int | None,
    ratelimit_minute: int | None,
    web_search_ratelimit_day: int | None,
    web_search_ratelimit_hour: int | None,
    web_search_ratelimit_minute: int | None,
):
    async def add_user():
        user = await UserManager.insert_user(
            name,
            apikey,
            userid,
            RateLimits(
                user_rate_limit_day=ratelimit_day,
                user_rate_limit_hour=ratelimit_hour,
                user_rate_limit_minute=ratelimit_minute,
                web_search_rate_limit_day=web_search_ratelimit_day,
                web_search_rate_limit_hour=web_search_ratelimit_hour,
                web_search_rate_limit_minute=web_search_ratelimit_minute,
            ),
        )
        json_user = json.dumps(
            {
                "userid": user.userid,
                "name": user.name,
                "apikey": user.apikey,
                "ratelimit_day": user.ratelimit_day,
                "ratelimit_hour": user.ratelimit_hour,
                "ratelimit_minute": user.ratelimit_minute,
            },
            indent=4,
        )

        print(json_user)

    asyncio.run(add_user())


if __name__ == "__main__":
    main()
