use redis::aio::ConnectionManager;

use nilai_domain::error::{NilaiError, NilaiResult};

pub struct RedisPool {
    conn: ConnectionManager,
    rate_limit_script_sha: String,
}

impl RedisPool {
    pub async fn new(redis_url: &str) -> NilaiResult<Self> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| NilaiError::Internal(format!("Redis client error: {}", e)))?;

        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| NilaiError::Internal(format!("Redis connection error: {}", e)))?;

        let mut setup_conn = conn.clone();

        let lua_script = r#"
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local expire_time = tonumber(ARGV[2])

local current = tonumber(redis.call('get', key) or "0")
if current > 0 then
    if current + 1 > limit then
        return redis.call("PTTL", key)
    else
        redis.call("INCR", key)
        return 0
    end
else
    if expire_time > 0 then
        redis.call("SET", key, 1, "px", expire_time)
    else
        redis.call("SET", key, 1)
    end
    return 0
end
"#;

        let sha: String = redis::cmd("SCRIPT")
            .arg("LOAD")
            .arg(lua_script)
            .query_async(&mut setup_conn)
            .await
            .map_err(|e| NilaiError::Internal(format!("Failed to load Lua script: {}", e)))?;

        Ok(Self {
            conn,
            rate_limit_script_sha: sha,
        })
    }

    pub fn connection(&self) -> ConnectionManager {
        self.conn.clone()
    }

    pub fn rate_limit_sha(&self) -> &str {
        &self.rate_limit_script_sha
    }
}
