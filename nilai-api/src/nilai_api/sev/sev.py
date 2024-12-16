import base64
import ctypes
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SEVGuest:
    def __init__(self):
        self.lib: Optional[ctypes.CDLL] = None
        self._load_library()

    def _load_library(self) -> None:
        try:
            lib_path = f"{os.path.dirname(os.path.abspath(__file__))}/libsevguest.so"
            if not os.path.exists(lib_path):
                logger.warning(f"SEV library not found at {lib_path}")
                return

            self.lib = ctypes.CDLL(lib_path)
            self._setup_library_functions()
        except Exception as e:
            logger.warning(f"Failed to load SEV library: {e}")
            self.lib = None

    def _setup_library_functions(self) -> None:
        if not self.lib:
            return

        self.lib.OpenDevice.restype = ctypes.c_int
        self.lib.GetQuoteProvider.restype = ctypes.c_int
        self.lib.Init.restype = ctypes.c_int
        self.lib.GetQuote.restype = ctypes.c_char_p
        self.lib.GetQuote.argtypes = [ctypes.c_char_p]
        self.lib.VerifyQuote.restype = ctypes.c_int
        self.lib.VerifyQuote.argtypes = [ctypes.c_char_p]
        self.lib.free.argtypes = [ctypes.c_char_p]

    def init(self) -> bool:
        """Initialize the device and quote provider."""
        if not self.lib:
            logger.warning("SEV library not loaded, running in mock mode")
            return True
        return self.lib.Init() == 0

    def get_quote(self, report_data: Optional[bytes] = None) -> str:
        """Get a quote using the report data."""
        if not self.lib:
            logger.warning("SEV library not loaded, returning mock quote")
            return base64.b64encode(b"mock_quote").decode("ascii")

        if report_data is None:
            report_data = bytes(64)

        if len(report_data) != 64:
            raise ValueError("Report data must be exactly 64 bytes")

        report_buffer = ctypes.create_string_buffer(report_data)
        quote_ptr = self.lib.GetQuote(report_buffer)

        if quote_ptr is None:
            raise RuntimeError("Failed to get quote")

        quote_str = ctypes.string_at(quote_ptr)
        return base64.b64encode(quote_str).decode("ascii")

    def verify_quote(self, quote: str) -> bool:
        """Verify the quote using the library's verification method."""
        if not self.lib:
            logger.warning(
                "SEV library not loaded, mock verification always returns True"
            )
            return True

        quote_bytes = base64.b64decode(quote.encode("ascii"))
        quote_buffer = ctypes.create_string_buffer(quote_bytes)
        return self.lib.VerifyQuote(quote_buffer) == 0


# Global instance
sev = SEVGuest()

if __name__ == "__main__":
    try:
        if sev.init():
            print("SEV guest device initialized successfully.")
            report_data = bytes([0] * 64)
            quote = sev.get_quote(report_data)
            print("Quote:", quote)

            if sev.verify_quote(quote):
                print("Quote verified successfully.")
            else:
                print("Quote verification failed.")
        else:
            print("Failed to initialize SEV guest device.")
    except Exception as e:
        print("Error:", e)
