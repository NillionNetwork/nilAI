package main

import (
	"fmt"

	"github.com/google/go-sev-guest/client"
	"github.com/google/go-sev-guest/verify"
)

var device client.Device
var quoteProvider client.QuoteProvider

func main() {
	var err error

	if device, err = client.OpenDevice(); err != nil {
		panic(fmt.Sprintf("failed to open device: %v", err))
	}

	if quoteProvider, err = client.GetQuoteProvider(); err != nil {
		panic(fmt.Sprintf("failed to get quote provider: %v", err))
	}

	// Use the device to get a quote.
	reportData := [64]byte{0}
	quote, err := client.GetQuoteProto(quoteProvider, reportData)
	if err != nil {
		panic(fmt.Sprintf("failed to get raw quote: %v", err))
	}

	quote.String()

	// Verify the quote.
	err = verify.SnpAttestation(quote, verify.DefaultOptions())
	if err != nil {
		panic(fmt.Sprintf("failed to verify quote: %v", err))
	}
}
