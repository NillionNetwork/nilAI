# Nilai Auth Server

This server acts as a delegation authority for Nillion User Compute (NUC) tokens, specifically for interacting with the Nilai API. It handles obtaining a root NUC token from a configured NilAuth instance, managing subscriptions on the Nillion Chain, and delegating compute capabilities to end-user public keys.

## Functionality

1.  **Wallet Initialization:** On startup (or first request), it initializes a Nilchain wallet using a hardcoded private key (for development purposes).
2.  **NilAuth Client:** Connects to a NilAuth instance specified in `NILAUTH_TRUSTED_ROOT_ISSUERS`.
3.  **Subscription Management:** Checks the Nilchain subscription status associated with its wallet. If not subscribed, it pays for the subscription using its wallet.
4.  **Root Token Retrieval:** Obtains a root NUC token from the NilAuth instance.
5.  **Delegation Endpoint (`/v1/delegate/`):**
    *   Accepts a POST request containing the end-user's public key (`user_public_key`).
    *   Validates the subscription and root token.
    *   Creates a new NUC token, extending the root token's capabilities.
    *   Sets the audience of the new token to the provided user public key.
    *   Authorizes the `nil ai generate` command.
    *   Signs the new token with its private key.
    *   Returns the delegated NUC token to the user.

## Prerequisites

*   A running NilAuth instance accessible at the URL(s) defined in the `NILAUTH_TRUSTED_ROOT_ISSUERS` environment variable (or configured within `nilai_auth_server/config.py`).
*   A running Nillion Chain node accessible via gRPC (currently hardcoded to `http://localhost:26649`).
*   The server's wallet must have sufficient `unil` tokens to pay for NilAuth subscriptions if needed.

## Running the Server

Use a ASGI server like Uvicorn:

```bash
cd nilai-auth/nilai-auth-server
uv run python3 src/nilai_auth_server/app.py
```

## Configuration

*   **Private Key:** Currently hardcoded within `app.py`. **This should be replaced with a secure key management solution for production.**
*   **Nilchain gRPC Endpoint:** Hardcoded to `http://localhost:26649` in `app.py`. Consider making this configurable.
*   **NilAuth Trusted Issuers:** Configured via `NILAUTH_TRUSTED_ROOT_ISSUERS` in `config.py`.

## API

### POST `/v1/delegate/`

*   **Request Body:**
    ```json
    {
      "user_public_key": "string (base64 encoded secp256k1 public key)"
    }
    ```
*   **Response Body:**
    ```json
    {
      "token": "string (NUC token envelope)"
    }
    ```
