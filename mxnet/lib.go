package mxnet

// #cgo CFLAGS: -g -O3
// #cgo LDFLAGS: -lmxnet
// #cgo darwin CFLAGS: -I/Users/abduld/frameworks/mxnet/include
// #cgo darwin LDFLAGS: -L/Users/abduld/frameworks/mxnet/lib
// #cgo linux CFLAGS: -I/opt/frameworks/mxnet/include -I/home/abduld/frameworks/mxnet/include
// #cgo linux LDFLAGS: -L/opt/frameworks/mxnet/lib -L/home/abduld/frameworks/mxnet/lib
import "C"
