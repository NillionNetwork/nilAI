# It defines the number of concurrent requests allowed for each model.
# At a same point of time, if all models are available,
# there can be 10 + 10 + 5 + 5 = 30 concurrent requests in the system
MODEL_CONCURRENT_RATE_LIMIT = {
    "meta-llama/Llama-3.2-1B-Instruct": 10,
    "meta-llama/Llama-3.2-3B-Instruct": 10,
    "meta-llama/Llama-3.1-8B-Instruct": 5,
    "cognitivecomputations/Dolphin3.0-Llama3.1-8B": 5,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 5,
    "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4": 5,
}

# It defines the number of requests allowed for each user for a given time frame.
# A user can make 10 requests per minute, 100 requests per hour, and 1000 requests per day.
# It follows a fixed window rate limiting algorithm:
# After you do your first request, the timer is set to 1 minute, 1 hour, and 1 day.
# You can make 9 more requests in the next 59 seconds, 99 in the next hour or 999 in the next day.
# After the time frame is over, the counter resets.
USER_RATE_LIMIT_MINUTE = 10
USER_RATE_LIMIT_HOUR = 100
USER_RATE_LIMIT_DAY = 1000
WEB_SEARCH_RATE_LIMIT_MINUTE = 1
WEB_SEARCH_RATE_LIMIT_HOUR = 3
WEB_SEARCH_RATE_LIMIT_DAY = 72
