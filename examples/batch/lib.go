package main

// #cgo linux CFLAGS: -I/usr/local/cuda/include
// #cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcuda
// #cgo pkg-config: protobuf
import "C"
