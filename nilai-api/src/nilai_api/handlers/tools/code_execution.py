from __future__ import annotations

import asyncio
import logging
from typing import Optional

from e2b_code_interpreter import Sandbox

logger = logging.getLogger(__name__)


def _run_in_sandbox_sync(code: str) -> str:
    logger.info("Starting code execution in sandbox")
    logger.debug(f"Code to execute: {code}")
    
    try:
        with Sandbox.create() as sandbox:
            logger.debug("Sandbox created successfully")
            execution = sandbox.run_code(code)
            logger.debug("Code execution completed")


            if execution.logs and execution.logs.stdout:
                output = "\n".join(execution.logs.stdout)
                logger.debug(f"Captured stdout output: {output}")
                return output
            
            if hasattr(execution, "text") and execution.text:
                logger.debug(f"Captured text output: {execution.text}")
                return execution.text

            try:
                result = getattr(execution, "result", None)
                if result is not None:
                    output = str(result)
                    logger.debug(f"Captured result output: {output}")
                    return output
            except Exception as e:
                logger.warning(f"Error getting execution result: {e}")
                pass
                
            logger.debug("No output captured from code execution")
            return ""
    except Exception as e:
        logger.error(f"Error executing code in sandbox: {e}")
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

