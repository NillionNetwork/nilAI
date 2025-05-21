# Nilai Auth Client

This client demonstrates the end-to-end process of authenticating with the Nilai API using Nillion User Compute (NUC) tokens obtained via the Nilai Auth Server.

## Functionality

1.  **Key Generation:** Generates a new secp256k1 private/public key pair for the user.
2.  **Request Delegation:** Sends the user's public key (base64 encoded) to the Nilai Auth Server (`/v1/delegate/` endpoint) to request a delegated NUC token.
3.  **Token Validation:** Validates the received delegated token against the public key of the NilAuth instance (acting as the root issuer).
4.  **Nilai Public Key Retrieval:** Fetches the public key of the target Nilai API instance (`/v1/public_key` endpoint).
5.  **Invocation Token Creation:** Creates an invocation NUC token by:
    *   Extending the previously obtained delegated token.
    *   Setting the audience to the Nilai API's public key.
    *   Signing the invocation token with the user's private key.
6.  **Invocation Token Validation:** Validates the created invocation token, ensuring it's correctly targeted at the Nilai API.
7.  **API Call:** Uses the `openai` library (configured with the Nilai API base URL) to make a chat completion request.
    *   The invocation token is passed as the `api_key` in the request header.
    *   The Nilai API verifies this token before processing the request.
8.  **Prints Response:** Outputs the response received from the Nilai API.

## Prerequisites

*   A running **Nilai Auth Server** (default: `localhost:8100`).
*   A running **Nilai API** instance (default: `localhost:8080`).
*   A running **NilAuth** node (default: `localhost:30921`).

## Running the Client

```bash
cd nilai-auth/nilai-auth-client
# Make sure dependencies are installed (e.g., using uv or pip)
python src/nilai_auth_client/main.py
```

## Configuration

Endpoints for the dependent services are currently hardcoded in `main.py`:

*   `SERVICE_ENDPOINT`: Nilai Auth Server (`localhost:8100`)
*   `NILAI_ENDPOINT`: Nilai API (`localhost:8080`)
*   `NILAUTH_ENDPOINT`: NilAuth Node (`localhost:30921`)

These could be made configurable via environment variables or command-line arguments if needed.
