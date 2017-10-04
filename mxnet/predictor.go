// Copyright 2016 go-mxnet-predictor Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mxnet

/*
// go preamble
#include <mxnet/c_predict_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
)

// predictor for inference
type Predictor struct {
	handle  C.PredictorHandle // C handle of predictor
	options *options.Options
}

// Create a Predictor
// go binding for MXPredCreate
// param symbol The JSON string of the symbol
// param params In-memory raw bytes of parameter ndarray file
// param device Device to run predictor
// param nodes An array of InputNode which stored the name and shape data of ndarray item
func CreatePredictor(opts ...options.Option) (*Predictor, error) {
	var (
		pc        *C.char
		shapeIdx  = []uint32{0}
		shapeData = []uint32{}
	)

	options := options.New(opts...)

	if len(options.Symbol()) == 0 {
		return nil, errors.New("invalid empty symbol")
	}
	if len(options.Weights()) == 0 {
		return nil, errors.New("invalid empty weights")
	}
	if len(options.InputNodes()) == 0 {
		return nil, errors.New("no input nodes found")
	}
	if len(options.Devices()) == 0 {
		return nil, errors.New("no devices defined")
	}

	symbol := options.Symbol()
	params := options.Weights()
	device := options.Devices()[0]
	nodes := options.InputNodes()

	// malloc a **char which like []string to store node keys
	keys := C.malloc(C.size_t(len(nodes)) * C.size_t(unsafe.Sizeof(pc))) // c gc
	j := 0
	for i := 0; i < len(nodes); i++ {
		// get memory address
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		// c gc
		*p = C.CString(nodes[i].Key())

		// shapeIdx for next node
		shapeIdx = append(shapeIdx, uint32(j+len(nodes[i].Shape())))
		j += len(nodes[i].Shape())
		// shape data for current node
		shapeData = append(shapeData, nodes[i].Shape()...)
	}

	var handle C.PredictorHandle

	err := C.MXPredCreate((*C.char)(unsafe.Pointer(&symbol[0])),
		unsafe.Pointer(&params[0]),
		C.int(len(params)),
		C.int(device.Type()),
		C.int(device.ID()),
		C.mx_uint(len(nodes)),
		(**C.char)(keys),
		(*C.mx_uint)(unsafe.Pointer(&shapeIdx[0])),
		(*C.mx_uint)(unsafe.Pointer(&shapeData[0])),
		&handle,
	)

	// free mem we created before return, go gc won't do that for us
	for i := 0; i < len(nodes); i++ {
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		C.free(unsafe.Pointer(*p))
	}
	C.free(unsafe.Pointer(keys))

	if err != 0 {
		return nil, GetLastError()
	}
	return &Predictor{handle: handle, options: options}, nil
}

// CreatePredictorPartial Creates a Predictor wich customized outputs [layer]
// go binding for MXPredCreate
// param symbol The JSON string of the symbol
// param params In-memory raw bytes of parameter ndarray file
// param device Device to run predictor
// param nodes An array of InputNode which stored the name and shape data of ndarray item
// param outputKey the name of the output layer/key
func CreatePredictorPartial(opts ...options.Option) (*Predictor, error) {
	var (
		pc        *C.char
		shapeIdx  = []uint32{0}
		shapeData = []uint32{}
	)

	options := options.New(opts...)

	if len(options.Symbol()) == 0 {
		return nil, errors.New("invalid empty symbol")
	}
	if len(options.Weights()) == 0 {
		return nil, errors.New("invalid empty weights")
	}
	if len(options.InputNodes()) == 0 {
		return nil, errors.New("no input nodes found")
	}
	if len(options.Devices()) == 0 {
		return nil, errors.New("no devices defined")
	}
	if options.OutputNode() == "" {
		return nil, errors.New("invalid empty outputNode")
	}

	symbol := options.Symbol()
	params := options.Weights()
	device := options.Devices()[0]
	nodes := options.InputNodes()
	outputKey := options.OutputNode()

	// malloc a **char which like []string to store node keys
	keys := C.malloc(C.size_t(len(nodes)) * C.size_t(unsafe.Sizeof(pc))) // c gc
	j := 0
	for i := 0; i < len(nodes); i++ {
		// get memory address
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		// c gc
		*p = C.CString(nodes[i].Key())

		// shapeIdx for next node
		shapeIdx = append(shapeIdx, uint32(j+len(nodes[i].Shape())))
		j += len(nodes[i].Shape())
		// shape data for current node
		shapeData = append(shapeData, nodes[i].Shape()...)
	}

	oKeys := C.malloc(C.size_t(len(nodes)) * C.size_t(unsafe.Sizeof(pc)))
	p := (**C.char)(unsafe.Pointer(uintptr(oKeys)))
	*p = C.CString(outputKey)

	var handle C.PredictorHandle

	success := C.MXPredCreatePartialOut((*C.char)(unsafe.Pointer(&symbol[0])),
		unsafe.Pointer(&params[0]),
		C.int(len(params)),
		C.int(device.Type()),
		C.int(device.ID()),
		C.mx_uint(len(nodes)),
		(**C.char)(keys),
		(*C.mx_uint)(unsafe.Pointer(&shapeIdx[0])),
		(*C.mx_uint)(unsafe.Pointer(&shapeData[0])),
		C.mx_uint(1),
		(**C.char)(oKeys),
		&handle,
	)

	// free mem we created before return, go gc won't do that for us
	for i := 0; i < len(nodes); i++ {
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(i)*unsafe.Sizeof(pc)))
		C.free(unsafe.Pointer(*p))
	}
	C.free(unsafe.Pointer(keys))

	if success != 0 {
		return nil, GetLastError()
	}
	return &Predictor{handle: handle, options: options}, nil
}

// set the input data of predictor
// go binding for MXPredSetInput
// param key The name of input node to set
// param data The float data to be set
func (s *Predictor) SetInput(key string, data []float32) error {
	// check input
	if data == nil || len(data) < 1 {
		return fmt.Errorf("intput data nil or empty")
	}

	batchSize := int64(s.options.BatchSize())
	if batchSize != 1 {
		var shape []uint32
		for _, inputNode := range s.options.InputNodes() {
			if inputNode.Key() == key {
				if len(inputNode.Shape()) == 3 {
					shape = inputNode.Shape()
				} else {
					shape = inputNode.Shape()[1:]
				}
			}
		}
		if len(shape) != 3 {
			return errors.New("invalid input shape")
		}
		dataLen := int64(len(data))
		shapeLen := int64(shape[0]) * int64(shape[1]) * int64(shape[2])
		inputCount := dataLen / shapeLen
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	k := C.CString(key)
	// free mem before return
	defer C.free(unsafe.Pointer(k))

	success := C.MXPredSetInput(s.handle,
		k,
		(*C.mx_float)(unsafe.Pointer(&data[0])),
		C.mx_uint(len(data)),
	)

	if success != 0 {
		return GetLastError()
	}
	return nil
}

// run a forward pass after SetInput
// go binding for MXPredForward
func (s *Predictor) Forward() error {
	success := C.MXPredForward(s.handle)
	if success != 0 {
		return GetLastError()
	}
	return nil
}

// get the shape of output node
// go binding for MXPredGetOutputShape
// param index The index of output node, set to 0 if there is only one output
func (s *Predictor) GetOutputShape(index uint32) ([]uint32, error) {
	var (
		shapeData *C.mx_uint
		shapeDim  C.mx_uint
	)
	success := C.MXPredGetOutputShape(s.handle,
		C.mx_uint(index),
		&shapeData,
		&shapeDim,
	)
	if success != 0 {
		return nil, GetLastError()
	}
	// c array to go
	shape := (*[1 << 32]uint32)(unsafe.Pointer(shapeData))[:shapeDim:shapeDim]
	return shape, nil
}

// get the output value of prediction
// go binding for MXPredGetOutput
// param index The index of output node, set to 0 if there is only one output
func (s *Predictor) GetOutput(index uint32) ([]float32, error) {
	shape, err := s.GetOutputShape(index)
	if err != nil {
		return nil, err
	}
	size := uint32(1)
	for _, v := range shape {
		size *= v
	}
	data := make([]float32, size)
	success := C.MXPredGetOutput(s.handle,
		C.mx_uint(index),
		(*C.mx_float)(unsafe.Pointer(&data[0])),
		C.mx_uint(size),
	)
	if success != 0 {
		return nil, GetLastError()
	}
	return data, nil
}

func (s *Predictor) Options() *options.Options {
	return s.options
}

// free this predictor's C handle
// go binding for MXPredFree
func (s *Predictor) Close() error {
	success := C.MXPredFree(s.handle)
	if success != 0 {
		return GetLastError()
	}
	return nil
}
