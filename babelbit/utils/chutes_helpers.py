from logging import getLogger
from json import loads, JSONDecodeError
from os import environ, chmod, stat
from stat import S_IEXEC
from asyncio import create_subprocess_exec, subprocess
from pathlib import Path
from contextlib import contextmanager
from random import Random

from jinja2 import Template
import petname

from babelbit.utils.settings import get_settings
from babelbit.utils.huggingface_helpers import get_huggingface_repo_name

from babelbit.utils.async_clients import get_async_client

logger = getLogger(__name__)


# ------------------------------ Internal helpers ------------------------------ #
def _log_chutes_failure(phase: str, returncode: int, lines: list[str]):
    """Log a concise but useful diagnostic block when a chutes CLI phase fails.

    Args:
        phase: build | deploy | warmup | share | delete etc.
        returncode: process return code
        lines: full collected stdout/stderr lines
    """
    # Keep the tail to avoid flooding logs while still providing context
    tail_n = 80
    tail = lines[-tail_n:]
    logger.error(
        "Chutes %s failed (exit=%s). Showing last %d/%d lines:\n%s",
        phase,
        returncode,
        len(tail),
        len(lines),
        "\n".join(tail),
    )
    # Try to surface any JSON block in the tail that might contain structured error info
    for ln in reversed(tail):
        if ln.strip().startswith("{") and '"error"' in ln:
            logger.error("Chutes %s structured error: %s", phase, ln.strip())
            break


@contextmanager
def temporary_chutes_config_file(prefix: str, delete: bool = True):
    settings = get_settings()
    tmp_path = settings.PATH_CHUTE_TEMPLATES / f"{prefix}.py"
    try:
        with open(tmp_path, "w+") as f:
            yield f, tmp_path
    finally:
        if delete:
            tmp_path.unlink(missing_ok=True)


def generate_nickname(key: str) -> str:
    petname.random = Random(int(key, 16))
    return petname.Generate(words=2, separator="-")


def get_chute_name(hf_revision: str) -> str:
    settings = get_settings()
    nickname = generate_nickname(key=hf_revision)
    logger.info(f"Hf Revision ({hf_revision}) -> Nickname ({nickname})")
    return f"{settings.HUGGINGFACE_USERNAME.replace('/','-')}-{nickname}".lower()


def guess_chute_slug(hf_revision: str) -> str:
    settings = get_settings()
    chute_username = settings.CHUTES_USERNAME.replace("_", "-")
    chute_name = get_chute_name(hf_revision=hf_revision)
    return f"{chute_username}-{chute_name}"


def render_chute_template(
    revision: str,
) -> Template:

    settings = get_settings()
    hf_repo_name = get_huggingface_repo_name()
    chute_name = get_chute_name(hf_revision=revision)

    path_template = settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_MAIN
    template = Template(path_template.read_text())
    rendered = template.render(
        predict_endpoint=settings.CHUTES_MINER_PREDICT_ENDPOINT,
        repo_name=hf_repo_name,
        revision=revision,
        chute_user=settings.CHUTES_USERNAME,
        chute_name=chute_name,
        schema_defs=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_SCHEMAS
        ).read_text(),
        setup_utils=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_SETUP_UTILS
        ).read_text(),
        load_utils=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_LOAD_UTILS
        ).read_text(),
        predict_utils=(
            settings.PATH_CHUTE_TEMPLATES / settings.FILENAME_CHUTE_PREDICT_UTILS
        ).read_text(),
    )
    return rendered


async def get_chute_slug_and_id(revision: str) -> tuple[str, str | None]:
    settings = get_settings()
    proc = await create_subprocess_exec(
        "chutes",
        "chutes",
        "get",
        get_chute_name(hf_revision=revision),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
    )
    out, _ = await proc.communicate()
    log = out.decode(errors="ignore")
    logger.info(log[-800:])
    if proc.returncode != 0:
        logger.error(log)
        raise ValueError("Chutes Query failed.")
    json_tail_of_log = "{" + "{".join(log.split("{")[1:])
    logger.info(json_tail_of_log)
    try:
        json_response = loads(json_tail_of_log)
    except JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from chutes output: {e}")
        json_response = {}
    slug = json_response.get("slug")
    chute_id = json_response.get("chute_id")
    if slug:
        logger.info(f"Slug found: {slug}\n Chute Id: {chute_id}")
        return slug, chute_id
    slug = guess_chute_slug(hf_revision=revision)
    logger.info(f"No Slug returned. Guessing Slug {slug}\n Chute Id: {chute_id}")
    return slug, chute_id


async def share_chute(chute_id: str) -> None:
    logger.info(
        "🤝 Temporary fix: Sharing private chute with the only testnet Vali to allow querying"
    )
    VALIDATOR_CHUTES_ID = "021f1aff-fa98-5f57-a5d2-727ba8bfe39d"

    settings = get_settings()
    proc = await create_subprocess_exec(
        "chutes",
        "share",
        "--chute-id",
        chute_id,
        "--user-id",
        VALIDATOR_CHUTES_ID,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")  # auto-confirm
        await proc.stdin.drain()
        proc.stdin.close()

    # Read and log output line by line as it appears
    assert proc.stdout is not None
    full_output = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="ignore").rstrip()
        full_output.append(decoded_line)
        logger.info(f"[chutes share] {decoded_line}")

    returncode = await proc.wait()
    if returncode != 0:
        raise ValueError("Chutes sharing failed.")


async def build_chute(path: Path) -> None:
    logger.info(
        "🚧 Building model on chutes... This may take a while. Please don't exit."
    )

    settings = get_settings()
    proc = await create_subprocess_exec(
        "chutes",
        "build",
        f"{path.stem}:chute",
        "--public",
        "--wait",
        "--debug",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
        cwd=str(path.parent),
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")  # auto-confirm
        await proc.stdin.drain()
        proc.stdin.close()

    # Read and log output line by line as it appears
    assert proc.stdout is not None
    full_output = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="ignore").rstrip()
        full_output.append(decoded_line)
        logger.info(f"[chutes build] {decoded_line}")

    returncode = await proc.wait()
    if returncode != 0:
        _log_chutes_failure("build", returncode, full_output)
        raise ValueError("Chutes building failed.")


async def warmup_chute(chute_id: str, timeout_minutes: int = 10, max_retries: int = 2) -> None:
    import asyncio
    import time
    
    logger.info("🧊🔥 Warming up chute: %s (timeout: %dm, retries: %d)", chute_id, timeout_minutes, max_retries)

    async def single_warmup_attempt(attempt: int) -> tuple[bool, list[str], int, int]:
        """Single warmup attempt. Returns (success, output_lines, returncode, connection_errors)"""
        if attempt > 1:
            logger.info("🔄 Warmup attempt %d/%d...", attempt, max_retries + 1)
        
        settings = get_settings()
        proc = await create_subprocess_exec(
            "chutes",
            "warmup",
            chute_id,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            env={
                **environ,
                "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
            },
        )
        
        # Auto-confirm any prompts
        if proc.stdin:
            proc.stdin.write(b"y\n")
            await proc.stdin.drain()
            proc.stdin.close()

        async def read_output():
            assert proc.stdout is not None
            full_output = []
            connection_errors = 0
            last_status_time = time.time()
            
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded_line = line.decode(errors="ignore").rstrip()
                full_output.append(decoded_line)
                logger.info(f"[chutes warmup] {decoded_line}")
                
                # Track network/connection errors
                if any(error in decoded_line.lower() for error in [
                    'clientpayloaderror', 'transferencodingerror', 'server disconnected',
                    'connection lost', 'not enough data', 'response payload is not completed',
                    'failed to connect', 'connection refused', 'network error'
                ]):
                    connection_errors += 1
                    logger.warning("⚠️ Network error detected (%d/3)", connection_errors)
                
                # Check for success
                if "Status: hot" in decoded_line or "warmup completed" in decoded_line.lower():
                    logger.info("✅ Chute warmup completed successfully!")
                    return full_output, True, connection_errors
                
                # Update last status time
                if "Status:" in decoded_line or "waiting for" in decoded_line.lower():
                    last_status_time = time.time()
                
                # Check for stuck state
                if time.time() - last_status_time > 120:  # 2 minutes without status
                    logger.warning("⏳ No status updates for 2+ minutes - chute may be stuck")
            
            return full_output, False, connection_errors

        try:
            full_output, success, connection_errors = await read_output()
            returncode = await proc.wait()
            
            return success, full_output, returncode, connection_errors
            
        except Exception as e:
            logger.warning("⚠️ Exception during warmup stream: %s", e)
            try:
                if proc.returncode is None:
                    proc.terminate()
                    await asyncio.wait_for(proc.wait(), timeout=5)
            except:
                proc.kill() if proc.returncode is None else None
            
            # Return failure with exception info
            return False, [f"Exception: {e}"], -1, 1
    
    # Main retry loop
    last_error = None
    for attempt in range(1, max_retries + 2):  # +1 for initial attempt
        try:
            # Run single attempt with timeout
            success, full_output, returncode, connection_errors = await asyncio.wait_for(
                single_warmup_attempt(attempt), 
                timeout=timeout_minutes * 60
            )
            
            if success:
                logger.info("✅ Chute warmup completed successfully")
                return
            
            # Analyze the failure
            error_msg = f"Chutes warmup failed with exit code {returncode}"
            
            # Check for specific error patterns
            full_output_text = '\n'.join(full_output)
            is_network_error = (
                connection_errors >= 2 or
                'ClientPayloadError' in full_output_text or
                'TransferEncodingError' in full_output_text or
                'server disconnected' in full_output_text.lower()
            )
            
            if is_network_error:
                error_msg += " (Network/API communication error)"
            elif 'waiting for instances' in full_output_text and len(full_output) > 20:
                error_msg += " (Chute provisioning timeout - may need longer warmup)"
            
            last_error = error_msg
            
            # Log the failure
            if full_output:
                _log_chutes_failure("warmup", returncode, full_output)
            
            # Decide whether to retry
            should_retry = (
                attempt <= max_retries and 
                (is_network_error or returncode in [-1, 1])  # Network errors or generic failures
            )
            
            if should_retry:
                wait_time = min(10 * attempt, 30)  # Progressive backoff: 10s, 20s, 30s
                logger.info("⏳ Waiting %ds before retry...", wait_time)
                await asyncio.sleep(wait_time)
                continue
            else:
                break
                
        except asyncio.TimeoutError:
            last_error = f"Chute warmup timed out after {timeout_minutes} minutes"
            if attempt <= max_retries:
                logger.warning("⏰ Timeout on attempt %d, retrying...", attempt)
                continue
            else:
                break
        except Exception as e:
            last_error = f"Unexpected error: {e}"
            if attempt <= max_retries:
                logger.warning("⚠️ Error on attempt %d: %s, retrying...", attempt, e)
                await asyncio.sleep(5)
                continue
            else:
                break
    
    # All attempts failed
    logger.error("❌ Chute warmup failed after %d attempts: %s", max_retries + 1, last_error)
    raise ValueError(f"Chute warmup failed after {max_retries + 1} attempts: {last_error}")


async def deploy_chute(path: Path) -> None:
    logger.info("🚀 Deploying model to chutes... This may take a moment..")

    settings = get_settings()
    proc = await create_subprocess_exec(
        "chutes",
        "deploy",
        f"{path.stem}:chute",
        # "--public",
        "--accept-fee",
        "--logging-enabled",
        "--debug",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
        cwd=str(path.parent),
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")  # auto-confirm
        await proc.stdin.drain()
        proc.stdin.close()

    assert proc.stdout is not None
    full_output = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="ignore").rstrip()
        full_output.append(decoded_line)
        logger.info(f"[chutes deploy] {decoded_line}")

    returncode = await proc.wait()
    if returncode != 0:
        _log_chutes_failure("deploy", returncode, full_output)
        raise ValueError("Chutes deployment failed.")


async def delete_chute(revision: str) -> None:
    logger.info(" Removing model from chutes..")

    settings = get_settings()
    _, chute_id = await get_chute_slug_and_id(revision=revision)
    proc = await create_subprocess_exec(
        "chutes",
        "chutes",
        "delete",
        chute_id,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env={
            **environ,
            "CHUTES_API_KEY": settings.CHUTES_API_KEY.get_secret_value(),
        },
    )
    if proc.stdin:
        proc.stdin.write(b"y\n")  # auto-confirm
        await proc.stdin.drain()
        proc.stdin.close()

    assert proc.stdout is not None
    full_output = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded_line = line.decode(errors="ignore").rstrip()
        full_output.append(decoded_line)
        logger.info(f"[chutes delete] {decoded_line}")

    returncode = await proc.wait()
    if returncode != 0:
        raise ValueError("Chutes delete failed.")


async def build_and_deploy_chute(path: Path) -> None:
    settings = get_settings()
    if not settings.CHUTES_API_KEY.get_secret_value():
        raise ValueError("CHUTES_API_KEY missing.")
    chmod(str(path), stat(str(path)).st_mode | S_IEXEC)
    await build_chute(path=path)
    await deploy_chute(path=path)


async def call_chutes_list_models() -> dict | str:
    settings = get_settings()
    session = await get_async_client()
    async with session.get(
        f"{settings.CHUTES_MINERS_ENDPOINT}/chutes/",
        headers={"Authorization": settings.CHUTES_API_KEY.get_secret_value()},
    ) as response:
        t = await response.text()
        logger.info(t)
        if response.status != 200:
            raise RuntimeError(f"{response.status}: {t[:200]}")
        try:
            return await response.json()
        except Exception as e:
            logger.error(e)
            return t


async def resolve_chute_id_and_slug(model_name: str) -> tuple[str, str]:
    """Jon: you can always query the /chutes/ endpoint (either list, or /chutes/{chute_id} on the API and extract the slug parameter for the subdomain"""
    chute_id = None
    chute_slug = None

    response = await call_chutes_list_models()
    if isinstance(response, dict) and "items" in response:
        chutes_list = response["items"]
    elif isinstance(response, list):
        chutes_list = response
    else:
        logger.error(response)
        chutes_list = []

    for ch in reversed(chutes_list):
        if any(ch.get(k) == model_name for k in ("model_name", "name", "readme")):
            chute_id = ch.get("chute_id") or ch.get("name") or ch.get("readme")
            chute_slug = ch.get("slug")
    if not chute_id or not chute_slug:
        raise Exception("Could not resolve chute_id/slug after deploy.")

    return chute_id, chute_slug


async def deploy_to_chutes(revision: str, skip: bool) -> tuple[str, str]:
    if skip:
        return None, None

    try:
        chute_deployment_script = render_chute_template(
            revision=revision,
        )
        with temporary_chutes_config_file(prefix="bb_chutes", delete=True) as (
            tmp,
            tmp_path,
        ):
            tmp.write(chute_deployment_script)
            tmp.flush()
            logger.info(f"Wrote Chute config: {tmp_path}")
            await build_and_deploy_chute(path=tmp_path)
            logger.info(f"Chute deployment successful")
        chute_slug, chute_id = await get_chute_slug_and_id(revision=revision)
        logger.info(f"Deployed chute_id={chute_id} slug={chute_slug}")
        return chute_id, chute_slug
    except Exception as e:
        logger.error(e)
        return None, None
