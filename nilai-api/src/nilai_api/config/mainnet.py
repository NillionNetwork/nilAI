AUTH_STRATEGY = "api_key"

# It defines the number of concurrent requests allowed for each model.
# At a same point of time, if all models are available,
# there can be 45 + 30 + 15 + 5 = 85 concurrent requests in the system
MODEL_CONCURRENT_RATE_LIMIT = {
    "meta-llama/Llama-3.2-1B-Instruct": 45,
    "meta-llama/Llama-3.2-3B-Instruct": 30,
    "meta-llama/Llama-3.1-8B-Instruct": 15,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 5,
    "meta-llama/Llama-3.2-1B-Instruct": 10,
    "watt-ai/watt-tool-8B": 10,
}

# It defines the number of requests allowed for each user for a given time frame.
# A user can make 10 requests per minute, 100 requests per hour, and 1000 requests per day.
# It follows a fixed window rate limiting algorithm. In this case, the rate limit is disabled.
USER_RATE_LIMIT_MINUTE = None
USER_RATE_LIMIT_HOUR = None
USER_RATE_LIMIT_DAY = None
