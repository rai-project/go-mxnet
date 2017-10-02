package mxnet

// #cgo amd64 pkg-config: mxnet
// #cgo linux,ppc64le CFLAGS: -I /home/carml/frameworks/mxnet/include -I /opt/mxnet/include
// #cgo linux,ppc64le LDFLAGS: -L /home/carml/frameworks/mxnet/lib -L /opt/mxnet/lib -lmxnet 
import "C"

