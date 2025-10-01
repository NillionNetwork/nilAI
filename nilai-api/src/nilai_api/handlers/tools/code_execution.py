from __future__ import annotations

import asyncio
import logging

from e2b_code_interpreter import Sandbox

logger = logging.getLogger(__name__)


def _run_in_sandbox_sync(code: str) -> str:
    """Execute Python code in an e2b sandbox and return the textual output or stdout if available."""
    try:
        with Sandbox.create() as sandbox:
            exec_ = sandbox.run_code(code)
            if exec_.text:
                return exec_.text
            if getattr(exec_, "logs", None) and getattr(exec_.logs, "stdout", None):
                return "\n".join(exec_.logs.stdout)
            return ""
    except Exception as e:
        logger.error("Error executing code in sandbox: %s", e)
        raise


async def execute_python(code: str) -> str:
    """Execute Python code in an e2b Code Interpreter sandbox and return the textual output.

    This function is async-safe and runs the blocking execution in a thread.
    """
    logger.info("Executing Python code asynchronously")
    try:
        result = await asyncio.to_thread(_run_in_sandbox_sync, code)
        logger.info("Python code execution completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in async Python code execution: {e}")
        raise
