import ctypes
from ctypes import c_char_p, c_int, POINTER, create_string_buffer

# Load the shared library
lib = ctypes.CDLL('./libsevguest.so')

# Define function signatures
lib.OpenDevice.restype = c_int
lib.GetQuoteProvider.restype = c_int
lib.Init.restype = c_int
lib.GetQuote.restype = c_char_p
lib.GetQuote.argtypes = [c_char_p]

# Python wrapper functions
def init():
    """Initialize the device and quote provider."""
    if lib.Init() != 0:
        raise RuntimeError("Failed to initialize SEV guest device and quote provider.")

def get_quote(report_data):
    """
    Get a quote using the report data.
    
    Args:
        report_data (bytes): A 64-byte array containing the report data.

    Returns:
        str: The generated quote as a string.
    """
    if len(report_data) != 64:
        raise ValueError("report_data must be exactly 64 bytes.")
    
    # Convert report data to a C-style string
    report_data_c = create_string_buffer(report_data)
    quote = lib.GetQuote(report_data_c)
    if not quote:
        raise RuntimeError("Failed to get quote.")
    return quote.decode('utf-8')

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
        print("Quote:", quote)
        
    except Exception as e:
        print("Error:", e)