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
	"context"
	"unsafe"

	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
)

// predictor for inference
type Predictor struct {
	handle  C.PredictorHandle // C handle of predictor
	options *options.Options
}

func prod(arry []int) int {
	accum := int(1)
	for _, e := range arry {
		accum *= int(e)
	}
	return accum
}

// Create a Predictor
// go binding for MXPredCreate
// param symbol The JSON string of the symbol
// param params In-memory raw bytes of parameter ndarray file
// param device Device to run predictor
// param nodes An array of InputNode which stored the name and shape data of ndarray item
func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

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

	if options.UsesGPU() && !nvidiasmi.HasGPU {
		return nil, errors.New("no GPU device")
	}

	symbol := options.Symbol()
	params := options.Weights()
	device := options.Devices()[0]
	nodes := options.InputNodes()

	// malloc a **char which like []string to store node keys
	keys := C.malloc(C.size_t(len(nodes)) * C.size_t(unsafe.Sizeof(pc))) // c gc
	jj := 0
	for ii, nd := range nodes {
		// get memory address
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(ii)*unsafe.Sizeof(pc)))
		// c gc
		*p = C.CString(nd.Key())

		shape := intSliceToUint32(nd.Shape())

		// shapeIdx for next node
		jj += len(shape)
		shapeIdx = append(shapeIdx, uint32(jj))
		// shape data for current node
		shapeData = append(shapeData, shape...)
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
	for ii := range nodes {
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(ii)*unsafe.Sizeof(pc)))
		C.free(unsafe.Pointer(*p))
	}
	C.free(unsafe.Pointer(keys))

	if err != 0 {
		return nil, GetLastError()
	}
	return &Predictor{handle: handle, options: options}, nil
}

// set the input data of predictor
// go binding for MXPredSetInput
// param key The name of input node to set
// param data The float data to be set
func (p *Predictor) SetInput(key string, data []float32) error {
	k := C.CString(key)
	// free mem before return
	defer C.free(unsafe.Pointer(k))

	success := C.MXPredSetInput(p.handle,
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
func (p *Predictor) Forward() error {
	success := C.MXPredForward(p.handle)
	if success != 0 {
		return GetLastError()
	}
	return nil
}

func (p *Predictor) Predict(ctx context.Context, data []float32) error {
	if data == nil || len(data) < 1 {
		return errors.New("intput data nil or empty")
	}

	inputNode := p.options.InputNodes()[0] // take the first input node
	if inputNode.Key() == "" {
		return errors.New("expecting a valid (non-empty) input layer name")
	}

	var shape []int
	if len(inputNode.Shape()) == 3 {
		shape = inputNode.Shape()
	} else {
		shape = inputNode.Shape()[1:]
	}

	shapeLen := prod(shape)
	dataLen := len(data)
	inputCount := dataLen / shapeLen
	batchSize := p.options.BatchSize()

	if batchSize > inputCount {
		padding := make([]float32, (batchSize-inputCount)*shapeLen)
		data = append(data, padding...)
	}

	err := p.SetInput(inputNode.Key(), data)
	if err != nil {
		return err
	}

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	err = p.Forward()
	if err != nil {
		return err
	}
	span.Finish()

	return nil
}

// get the shape of output node
// go binding for MXPredGetOutputShape
// param index The index of output node, set to 0 if there is only one output
func (p *Predictor) GetOutputShape(index int) ([]int, error) {
	var (
		shapeData *C.mx_uint = nil
		shapeDim  C.mx_uint  = 0
	)
	success := C.MXPredGetOutputShape(p.handle,
		C.mx_uint(index),
		&shapeData,
		&shapeDim,
	)
	if success != 0 {
		return nil, GetLastError()
	}
	// c array to go
	shape := (*[1 << 32]C.mx_uint)(unsafe.Pointer(shapeData))[:shapeDim:shapeDim]
	res := make([]int, shapeDim)
	for ii, s := range shape {
		res[ii] = int(s)
	}
	return res, nil
}

// get the output of the prediction
// index is the index of the output node, set to 0 assuming there is only one output
func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]float32, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_prediction_output")
	defer span.Finish()

	index := 0

	shape, err := p.GetOutputShape(index)
	if err != nil {
		return nil, err
	}

	size := prod(shape)
	probabilities := make([]float32, size)
	success := C.MXPredGetOutput(p.handle,
		C.mx_uint(index),
		(*C.mx_float)(unsafe.Pointer(&probabilities[0])),
		C.mx_uint(size),
	)
	if success != 0 {
		return nil, GetLastError()
	}

	return probabilities, nil
}

// free this predictor's C handle
// go binding for MXPredFree
func (p *Predictor) Close() error {
	success := C.MXPredFree(p.handle)
	if success != 0 {
		return GetLastError()
	}
	return nil
}
