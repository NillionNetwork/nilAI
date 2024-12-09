import os
import base64
import ctypes
from ctypes import c_char_p, c_int, c_void_p, create_string_buffer

# Load the shared library
lib = ctypes.CDLL(f"{os.path.dirname(os.path.abspath(__file__))}/libsevguest.so")

# OpenDevice
lib.OpenDevice.restype = c_int

# GetQuoteProvider
lib.GetQuoteProvider.restype = c_int

# Init
lib.Init.restype = c_int

# GetQuote
lib.GetQuote.restype = c_char_p
lib.GetQuote.argtypes = [c_char_p]

# VerifyQuote
lib.VerifyQuote.restype = c_int
lib.VerifyQuote.argtypes = [c_char_p]

lib.free.argtypes = [c_char_p]

# Python wrapper functions
def init():
    """Initialize the device and quote provider."""
    if lib.Init() != 0:
        raise RuntimeError("Failed to initialize SEV guest device and quote provider.")


def get_quote(report_data=None) -> str:
    """
    Get a quote using the report data.

    Args:
        report_data (bytes, optional): 64-byte report data.
                                        Defaults to 64 zero bytes.

    Returns:
        str: The quote as a string
    """
    # Use 64 zero bytes if no report data provided
    if report_data is None:
        report_data = bytes(64)

    # Validate report data
    if len(report_data) != 64:
        raise ValueError("Report data must be exactly 64 bytes")

    # Create a buffer from the report data
    report_buffer = create_string_buffer(report_data)

    # Get the quote
    quote_ptr = lib.GetQuote(report_buffer)
    quote_str = ctypes.string_at(quote_ptr)

    # We should be freeing the quote, but it turns out it raises an error.
    # lib.free(quote_ptr)
    # Check if quote retrieval failed
    if quote_ptr is None:
        raise RuntimeError("Failed to get quote")

    # Convert quote to Python string
    quote = base64.b64encode(quote_str)
    return quote.decode("ascii")


def verify_quote(quote: str) -> bool:
    """
    Verify the quote using the library's verification method.

    Args:
        quote (str): The quote to verify

    Returns:
        bool: True if quote is verified, False otherwise
    """
    # Ensure quote is a string
    if not isinstance(quote, str):
        quote = str(quote)

    # Convert to bytes
    quote_bytes = base64.b64decode(quote.encode("ascii"))
    quote_buffer = create_string_buffer(quote_bytes)

    # Verify quote
    result = lib.VerifyQuote(quote_buffer)
    return result == 0


# Example usage
if __name__ == "__main__":
    try:
        # Initialize the device and quote provider
        init()
        print("SEV guest device initialized successfully.")

        # Create a 64-byte report data array (all zeros for simplicity)
        report_data = bytes([0] * 64)

        # Get the quote
        quote = get_quote(report_data)
        print(type(quote))
        print("Quote:", quote)

        if verify_quote(quote):
            print("Quote verified successfully.")
        else:
            print("Quote verification failed.")
    except Exception as e:
        print("Error:", e)
