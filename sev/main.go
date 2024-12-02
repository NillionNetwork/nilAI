package main

import (
	"C"
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
	// Transform the reportData from C.char to []byte
	reportDataBytes := [64]byte{0}
	for i := 0; i < 64; i++ {
		reportDataBytes[i] = byte(*(*C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(reportData)) + uintptr(i))))
	}

	// Use the device to get a quote.
	//quote, err := quoteProvider.GetRawQuote(*reportDataBytes)
	quote, err := client.GetQuoteProto(quoteProvider, reportDataBytes)
	if err != nil {
		return C.CString(err.Error())
	}
	return C.CString(quote.String())
}

//export VerifyQuote
func VerifyQuote(quoteStr *C.char) int {
	// Change the quoteStr from C.char to string
	quote := []byte(C.GoString(quoteStr))
	err := verify.SnpAttestation(&quote, verify.DefaultOptions())
	if err != nil {
		fmt.Printf("failed to verify quote: %v\n", err)
		return -1
	}
	return 0
}

func main() {}
