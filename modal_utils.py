import time
from typing import Any
import modal

_function_cache: dict[tuple[str, str], modal.Function] = {}


def _get_modal_function(app_name: str, function_name: str) -> modal.Function:
    key = (app_name, function_name)
    fn = _function_cache.get(key)
    if fn is None:
        fn = modal.Function.from_name(app_name, function_name)
        _function_cache[key] = fn
    return fn


def run_modal_job(
    app_name: str,
    function_name: str,
    timeout_minutes: int = 10,
    poll_delay_sec: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    fn = _get_modal_function(app_name, function_name)
    call = fn.spawn(**kwargs)

    print(f"[{app_name}] Modal started.")

    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    last_minute_logged = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"[{app_name}] Timeout after {timeout_minutes} minutes.")

        try:
            output = call.get(timeout=0)
            print(f"[{app_name}] Modal finished successfully.")
            return {
                "status": "COMPLETED",
                "output": output,
                "elapsed_seconds": round(elapsed, 2),
            }
        except TimeoutError:
            pass

        elapsed_minutes = int(elapsed // 60)
        if elapsed_minutes >= 1 and elapsed_minutes > last_minute_logged:
            last_minute_logged = elapsed_minutes
            print(f"[{app_name}] Modal still waiting... ({elapsed_minutes} min elapsed)")

        time.sleep(poll_delay_sec)