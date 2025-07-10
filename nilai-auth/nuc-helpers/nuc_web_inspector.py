import streamlit as st

# Assuming nuc library is installed and available in the environment
try:
    from nuc.envelope import NucTokenEnvelope
    from nuc.token import NucToken, InvocationBody, Command, Did
    from nuc.validate import (
        NucTokenValidator,
        ValidationParameters,
        InvocationRequirement,
    )
    from nuc.nilauth import NilauthClient
    from nuc_helpers.helpers import get_nilai_public_key  # Reuse this helper
except ImportError:
    st.error(
        "Error: The 'nuc' library is not installed. Please install it to use this tool."
    )
    st.info("You might need to run: pip install nuc")
    st.stop()  # Stop the Streamlit app if nuc is not found


def display_token_details_streamlit(token_data, prefix="", is_proof=False):
    """Recursively displays the fields of a NUC token or a proof token in Streamlit."""
    if isinstance(token_data, NucToken):
        st.subheader(f"{prefix}{'Proof Token' if is_proof else 'NUC Token'} Details")
        st.write(f"**Issuer:** {token_data.issuer if token_data.issuer else 'N/A'}")
        st.write(
            f"**Audience:** {token_data.audience if token_data.audience else 'N/A'}"
        )
        st.write(f"**Subject:** {token_data.subject if token_data.subject else 'N/A'}")
        st.write(f"**Command:** {token_data.command if token_data.command else 'N/A'}")
        st.write(
            f"**Not Before:** {token_data.not_before.isoformat() if token_data.not_before else 'N/A'}"
        )
        st.write(
            f"**Expires At:** {token_data.expires_at.isoformat() if token_data.expires_at else 'N/A'}"
        )
        st.write(f"**Nonce:** {token_data.nonce.hex() if token_data.nonce else 'N/A'}")
        st.write(
            f"**Raw Proofs (bytes):** {[p.hex() for p in token_data.proofs] if token_data.proofs else 'N/A'}"
        )

        if token_data.meta:
            st.write("**Meta:**")
            st.json(token_data.meta)
        else:
            st.write("**Meta:** N/A")

        if token_data.body and isinstance(token_data.body, InvocationBody):
            if token_data.body.args:
                st.write("**Invocation Args:**")
                st.json(token_data.body.args)
            else:
                st.write("**Invocation Args:** N/A")
        # Add other body types if necessary, e.g., DelegationBody
        # elif token_data.body and isinstance(token_data.body, DelegationBody):
        #     st.write("**Delegation Body Details:** ...")
    elif isinstance(token_data, dict):
        for key, value in token_data.items():
            if isinstance(value, (dict, list)):
                with st.expander(f"{prefix}{key}:"):
                    display_token_details_streamlit(value, prefix + "  ", is_proof)
            else:
                st.write(f"{prefix}**{key}:** {value}")
    else:
        st.write(f"{prefix}{token_data}")


st.set_page_config(layout="wide", page_title="NUC Token Web Inspector")

st.title("NUC Token Web Inspector")

st.sidebar.header("Configuration")
nilauth_url = st.sidebar.text_input(
    "NilAuth URL", "https://nilauth.sandbox.app-cluster.sandbox.nilogy.xyz"
)
nilai_api_url = st.sidebar.text_input(
    "Nilai API URL", "https://nilai-a779.nillion.network/nuc"
)

st.sidebar.header("How to Use")
st.sidebar.markdown(
    "Paste your NUC token string into the text area on the left and click 'Inspect NUC' to view its details. "
    "The tool will display the main token's fields and recursively show details of any proofs within the token chain."
)
st.sidebar.markdown(
    "Click 'Validate NUC' to perform a validation check against the configured NilAuth and Nilai API endpoints."
    "This requires the NilAuth and Nilai API services to be running and accessible from where you run this app."
)
st.sidebar.markdown(
    "**Note:** Ensure the `nuc` library is installed in your Python environment (`pip install nuc`)."
)
st.sidebar.markdown(
    "You can generate sample NUC tokens using the `b2b2b2c_test()` function in `nilai-auth/nuc-helpers/src/nuc_helpers/main.py`."
)

input_col, display_col = st.columns([1, 2])

with input_col:
    st.header("Input NUC Token")
    nuc_string = st.text_area(
        "NUC Token String",
        height=300,
        placeholder="Paste your NUC token string here...",
        label_visibility="hidden",
    )
    inspect_button = st.button("Inspect NUC")
    validate_button = st.button("Validate NUC")  # New button

with display_col:
    st.header("Inspection Results")

    if inspect_button:
        if nuc_string:
            with st.spinner("Parsing NUC token..."):
                try:
                    envelope = NucTokenEnvelope.parse(nuc_string)
                    st.success("NUC Token parsed successfully!")
                    st.markdown("--- ")
                    st.subheader("Main Token Details")
                    display_token_details_streamlit(
                        envelope.token.token, is_proof=False
                    )

                    if envelope.proofs:
                        st.markdown("--- ")
                        st.subheader("Proofs (Recursive Chain)")
                        for i, proof in enumerate(envelope.proofs):
                            with st.expander(
                                f"Proof {i + 1} (Signature: {proof.signature.hex()})"
                            ):
                                if proof.token:
                                    display_token_details_streamlit(
                                        proof.token, is_proof=True
                                    )
                                else:
                                    st.warning("Proof token details not available.")
                    else:
                        st.info("No proofs found in this NUC.")

                except Exception as e:
                    st.error(f"Error parsing NUC token: {e}")
                    st.warning(
                        "Please ensure the provided string is a valid NUC token."
                    )
        else:
            st.warning("Please enter a NUC token string to inspect.")
    elif validate_button:  # New validation logic
        if nuc_string:
            with st.spinner("Validating NUC token..."):
                try:
                    envelope = NucTokenEnvelope.parse(nuc_string)

                    # 1. Get NilAuth Public Key (Trust Anchor)
                    nilauth_client = NilauthClient(nilauth_url)
                    nilauth_public_key = Did(
                        nilauth_client.about().public_key.serialize()
                    )
                    validator = NucTokenValidator([nilauth_public_key])

                    # 2. Get Nilai API Public Key for Audience Validation
                    nilai_public_key = get_nilai_public_key(nilai_api_url)
                    if not nilai_public_key:
                        st.error(
                            "Could not retrieve Nilai API public key. Cannot perform audience validation."
                        )
                        st.stop()  # Stop validation if key is missing

                    # 3. Define Validation Parameters
                    validation_parameters = ValidationParameters.default()
                    validation_parameters.token_requirements = InvocationRequirement(
                        audience=Did(nilai_public_key.serialize())
                    )

                    # 4. Check for Nilai Subcommand (from nuc.py)
                    NILAI_BASE_COMMAND: Command = Command.parse("/nil/ai")
                    command: Command = envelope.token.token.command
                    if not command.is_attenuation_of(NILAI_BASE_COMMAND):
                        st.error(
                            f"Validation Failed: NUC token command '{command}' is not an attenuation of '/nil/ai'."
                        )
                        st.stop()

                    # 5. Perform Validation
                    validator.validate(
                        envelope, context={}, parameters=validation_parameters
                    )

                    st.success("NUC Token is VALID!")
                    st.write(f"**Issuer:** {envelope.token.token.issuer}")
                    st.write(f"**Subject:** {envelope.token.token.subject}")
                    st.write(f"**Audience:** {envelope.token.token.audience}")
                    st.write(f"**Command:** {envelope.token.token.command}")

                except Exception as e:
                    st.error(f"NUC Token Validation FAILED: {e}")
                    st.warning(
                        "Please check the NUC token, configuration URLs, and ensure services are running."
                    )
        else:
            st.warning("Please enter a NUC token string to validate.")
    else:
        st.info(
            "Enter a NUC token and click 'Inspect NUC' or 'Validate NUC' to see details."
        )
