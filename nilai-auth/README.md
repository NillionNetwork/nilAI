# Example: nilAuth services.

# nilAuth Services

This repository contains two main services:

## nilai-auth-server

This server acts as a delegation authority for Nillion User Compute (NUC) tokens, specifically for interacting with the Nilai API. It handles obtaining a root NUC token from a configured NilAuth instance, managing subscriptions on the Nillion Chain, and delegating compute capabilities to end-user public keys. See `nilai-auth/nilai-auth-server/README.md` for more details.

## nilai-auth-client

This client demonstrates the end-to-end process of authenticating with the Nilai API using Nillion User Compute (NUC) tokens obtained via the Nilai Auth Server. See `nilai-auth/nilai-auth-client/README.md` for more details.
