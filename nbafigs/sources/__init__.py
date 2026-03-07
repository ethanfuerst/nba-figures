import time

from requests.exceptions import ConnectionError, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

nba_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((ConnectionError, Timeout)),
    reraise=True,
)


def nba_api_call(func, *args, **kwargs):
    '''Call an nba_api endpoint with retry and rate-limit delay.'''
    result = nba_api_retry(func)(*args, **kwargs)
    time.sleep(0.6)
    return result
