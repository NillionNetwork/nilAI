package main

// #include <stdio.h>
// #include <stdlib.h>
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/google/go-sev-guest/client"
	"github.com/google/go-sev-guest/verify"
)

var device client.Device
var quoteProvider client.QuoteProvider

//export OpenDevice
func OpenDevice() int {
	var err error
	if device, err = client.OpenDevice(); err != nil {
		fmt.Printf("failed to open device: %v\n", err)
		return -1
	}
	return 0
}

//export GetQuoteProvider
func GetQuoteProvider() int {
	var err error
	if quoteProvider, err = client.GetQuoteProvider(); err != nil {
		fmt.Printf("failed to get quote provider: %v\n", err)
		return -1
	}
	return 0
}

//export Init
func Init() int {
	if OpenDevice() != 0 || GetQuoteProvider() != 0 {
		return -1
	}
	return 0
}

//export GetQuote
func GetQuote(reportData *C.char) *C.char {
	if reportData == nil {
		return nil
	}

	// Convert reportData to a Go byte slice.
	var reportDataBytes [64]byte
	for i := 0; i < 64; i++ {
		reportDataBytes[i] = byte(C.char(*(*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(reportData)) + uintptr(i)))))
	}

	// Get the quote using the provided QuoteProvider.
	quote, err := client.GetQuoteProto(quoteProvider, reportDataBytes)
	if err != nil {
		return nil
	}
	result := C.CString(quote.GetReport().String())
	//result := C.CString(quote.String())
	return result
}

//export VerifyQuote
func VerifyQuote(quoteStr *C.char) int {
	// Change the quoteStr from C.char to string
	report := []byte(C.GoString(quoteStr))

	err := verify.RawSnpReport(report, verify.DefaultOptions())
	if err != nil {
		fmt.Printf("failed to verify quote: %v\n", err)
		return -1
	}
	return 0
}

func test() {
	Init()

	// Transform the reportData from C.char to []byte
	reportDataBytes := [64]byte{0}

	quote2, err := client.GetQuoteProto(quoteProvider, reportDataBytes)
	if err != nil {
		panic("B")
	}

	err = verify.SnpReport(quote2.GetReport(), verify.DefaultOptions())
	if err != nil {
		fmt.Printf("failed to verify report: %v\n", err)
	}
	// Use the device to get a quote.
	//quote, err := quoteProvider.GetRawQuote(*reportDataBytes)
	quote, err := client.GetQuoteProto(quoteProvider, reportDataBytes)
	if err != nil {
		panic("A")
	}
	quote_bytes := []byte(quote.String())
	err = verify.RawSnpReport(quote_bytes, verify.DefaultOptions())

	if err != nil {
		panic("Failed to verify quote")
	}
}
func main() {}
