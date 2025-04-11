package main

import (
	"fmt"

	"github.com/google/go-tdx-guest/client"
	"github.com/google/go-tdx-guest/verify"

	"github.com/google/go-tdx-guest/rtmr"
)

var device client.Device
var quoteProvider client.QuoteProvider

func main() {
	// Choose a mock device or a real device depending on the --tdx_guest_device_path flag.
	var err error

	if device, err = client.OpenDevice(); err != nil {
		panic(fmt.Sprintf("failed to open device: %v", err))
	}

	if quoteProvider, err = client.GetQuoteProvider(); err != nil {
		panic(fmt.Sprintf("failed to get quote provider: %v", err))
	}

	// Use the device to get a quote.
	reportData := [64]byte{0}
	quote, err := client.GetQuote(quoteProvider, reportData)
	if err != nil {
		panic(fmt.Sprintf("failed to get raw quote: %v", err))
	}

	// Close the device.
	if err := device.Close(); err != nil {
		panic(fmt.Sprintf("failed to close device: %v", err))
	}

	// Verify the quote.
	err = verify.TdxQuote(quote, &verify.Options{})
	if err != nil {
		panic(fmt.Sprintf("failed to verify quote: %v", err))
	}

	// This is a mock digest.
	// It should be the docker image hash.
	digest := [64]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	// Extend the digest to the rtmr 3
	rtmr.ExtendDigest(3, digest[:])

	// Get the RTM report.
	rtmReport, err := rtmr.GetRtmrsFromTdQuote(quote)
	if err != nil {
		panic(fmt.Sprintf("failed to get RTM report: %v", err))
	}

	_ = rtmReport

}
