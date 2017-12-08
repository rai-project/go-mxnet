package mxnet

// #cgo CFLAGS: -I/opt/mxnet/include
// #cgo LDFLAGS: -L/opt/mxnet/lib -lmxnet
// #cgo linux CFLAGS: -I /home/carml/frameworks/mxnet/include -I /opt/mxnet/include  -I /opt/frameworks/mxnet/include
// #cgo linux LDFLAGS: -L /home/carml/frameworks/mxnet/lib -L /opt/mxnet/lib -L /opt/frameworks/mxnet/lib -lmxnet
import "C"
