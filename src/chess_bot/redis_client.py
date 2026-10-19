import redis
import os
from dotenv import load_dotenv


load_dotenv()

REDIS_HOST: str = os.environ.get("REDIS_HOST")
REDIS_PORT: str = os.environ.get("REDIS_PORT")


def redis_client(key: str = None, value: str = None, method: str = "get") -> None:
    with redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True) as r:
        return getattr(r, method)(key) if method=="get" or method=="delete" \
            else getattr(r, method)(key, value)
