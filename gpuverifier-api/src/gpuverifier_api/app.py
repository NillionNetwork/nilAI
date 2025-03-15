# simple_gpu_attestation_service.py
from fastapi import FastAPI, HTTPException
import asyncio
import secrets
import json
import base64
import logging
from typing import Dict, Any

# Import the verifier functions
from verifier.cc_admin import collect_gpu_evidence, attest

app = FastAPI(title="Simple GPU Attestation Service")

@app.get("/")
async def root():
    return {"status": "GPU Attestation Service is running"}

@app.post("/attest-gpu", response_model=Dict[str, str])
async def get_gpu_attestation():
    """Get GPU attestation and return the quote"""
    try:
        # Run the attestation in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, gpu_attestation)
        
        return {"gpu_quote": result}
    except Exception as e:
        logging.error("Error in attestation endpoint: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Error getting GPU attestation: {str(e)}")

def gpu_attestation() -> str:
    # Check if GPU is available
    try:
        nonce = secrets.token_bytes(32).hex()
        arguments_as_dictionary = {
            "nonce": nonce,
            "verbose": False,
            "test_no_gpu": False,
            "rim_root_cert": None,
            "rim_service_url": None,
            "ocsp_service_url": None,
            "ocsp_attestation_settings": "default",
            "allow_hold_cert": None,
            "ocsp_validity_extension": None,
            "ocsp_cert_revocation_extension_device": None,
            "ocsp_cert_revocation_extension_driver_rim": None,
            "ocsp_cert_revocation_extension_vbios_rim": None,
        }
        evidence_list = collect_gpu_evidence(
            nonce,
        )
        result, jwt_token = attest(arguments_as_dictionary, nonce, evidence_list)
        gpu_quote = base64.b64encode(
            json.dumps({"result": result, "jwt_token": jwt_token}).encode()
        ).decode()
        return gpu_quote
    except Exception as e:
        logging.error("Could not attest GPU: %s", e)
        # Return empty string or specific error indicator
        return ""

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
