import argparse
import json

# Assuming nuc library is installed and available in the environment
try:
    from nuc.envelope import NucTokenEnvelope
    from nuc.token import NucToken, InvocationBody
except ImportError:
    print(
        "Error: The 'nuc' library is not installed. Please install it to use this tool."
    )
    print("You might need to run: pip install nuc")
    exit(1)


def display_token_details(token_data, prefix="", is_proof=False):
    """Recursively displays the fields of a NUC token or a proof token."""
    indent = "  " * len(prefix)
    print(f"{indent}{prefix}Type: {'Proof Token' if is_proof else 'NUC Token'}")

    if isinstance(token_data, NucToken):
        print(
            f"{indent}{prefix}  Issuer: {token_data.issuer if token_data.issuer else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Audience: {token_data.audience if token_data.audience else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Subject: {token_data.subject if token_data.subject else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Command: {token_data.command if token_data.command else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Not Before: {token_data.not_before.isoformat() if token_data.not_before else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Expires At: {token_data.expires_at.isoformat() if token_data.expires_at else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Meta: {json.dumps(token_data.meta, indent=2).replace('\n', '\n' + indent + '    ') if token_data.meta else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Nonce: {token_data.nonce.hex() if token_data.nonce else 'N/A'}"
        )
        print(
            f"{indent}{prefix}  Raw Proofs (bytes): {[p.hex() for p in token_data.proofs] if token_data.proofs else 'N/A'}"
        )
        if token_data.body and isinstance(token_data.body, InvocationBody):
            print(
                f"{indent}{prefix}  Invocation Args: {json.dumps(token_data.body.args, indent=2).replace('\n', '\n' + indent + '    ') if token_data.body.args else 'N/A'}"
            )
        # Add other body types if necessary, e.g., DelegationBody
        # elif token_data.body and isinstance(token_data.body, DelegationBody):
        #     print(f"{indent}{prefix}  Delegation Body Details: ...")
    elif isinstance(token_data, dict):
        for key, value in token_data.items():
            if isinstance(value, (dict, list)):
                print(f"{indent}{prefix}  {key}:")
                display_token_details(value, prefix + "  ", is_proof)
            else:
                print(f"{indent}{prefix}  {key}: {value}")
    else:
        print(f"{indent}{prefix}  {token_data}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect NUC (Nilchain Universal Credential) tokens."
    )
    parser.add_argument("nuc_string", help="The NUC token string to inspect.")
    args = parser.parse_args()

    try:
        envelope = NucTokenEnvelope.parse(args.nuc_string)

        print("--- NUC Token Details ---")
        display_token_details(envelope.token.token, is_proof=False)

        if envelope.proofs:
            print("\n--- Proofs (Recursive) ---")
            for i, proof in enumerate(envelope.proofs):
                print(f"Proof {i + 1}:")
                print(f"  Signature: {proof.signature.hex()}")
                if proof.token:
                    display_token_details(proof.token, prefix="  ", is_proof=True)
                else:
                    print("  (Proof token details not available)")
        else:
            print("\nNo proofs found in this NUC.")

    except Exception as e:
        print(f"Error parsing NUC token: {e}")
        print("Please ensure the provided string is a valid NUC token.")


if __name__ == "__main__":
    main()
