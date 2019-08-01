package mxnet

/*
#include <mxnet/c_predict_api.h>
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"runtime"
  "unsafe"
  "path/filepath"
  "strings"
  
  opentracing "github.com/opentracing/opentracing-go"
	gotensor "gorgonia.org/tensor"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
  "github.com/rai-project/tracer"
  cupti "github.com/rai-project/go-cupti"
)

// predictor for inference
type Predictor struct {
	handle  C.PredictorHandle // C handle of predictor
  options *options.Options
  cu        *cupti.CUPTI
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
	if len(options.Graph()) == 0 {
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

	if options.DisableFrameworkAutoTuning() {
		disableFrameworkAutoTuning()
	}

	symbol := options.Graph()
	params := options.Weights()
	device := options.Devices()[0]
	nodes := options.InputNodes()

	if options.UsesGPU() && !nvidiasmi.HasGPU {
		return nil, errors.New("no GPU device")
	}

	// malloc a **char which like []string to store node keys
	keys := C.malloc(C.size_t(len(nodes)) * C.size_t(unsafe.Sizeof(pc))) // c gc
	jj := 0
	for ii, nd := range nodes {
		// get memory address
		p := (**C.char)(unsafe.Pointer(uintptr(keys) + uintptr(ii)*unsafe.Sizeof(pc)))
		// c gc
		*p = C.CString(nd.Key)
		shape := intSliceToUint32(nd.Shape)
		// shapeIdx for next node
		jj += len(shape)
		shapeIdx = append(shapeIdx, uint32(jj))
		// shape data for current node
		shapeData = append(shapeData, shape...)
	}

	var handle C.PredictorHandle

	err := C.MXPredCreate(
		(*C.char)(unsafe.Pointer(&symbol[0])),
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

	pred := &Predictor{handle: handle, options: options}

	runtime.SetFinalizer(pred, (*Predictor).finalizer)

	return pred, nil
}

func (p *Predictor) GetOptions() *options.Options {
  return p.options
}

// set the input data of predictor
// go binding for MXPredSetInput
// param key The name of input node to set
// param data The float data to be set
func (p *Predictor) SetInput(key string, input *gotensor.Dense) error {
	k := C.CString(key)
	// free mem before return
	defer C.free(unsafe.Pointer(k))

	success := C.MXPredSetInput(
		p.handle,
		k,
		(*C.mx_float)(input.Pointer()),
		C.mx_uint(input.Size()),
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
	return WaitAll()
}

func (p *Predictor) Predict(ctx context.Context, data []*gotensor.Dense) error {
  if len(data) == 0 {
		return errors.New("intput data nil or empty")
	}

	for ii, inputNode := range p.options.InputNodes() {
		if inputNode.Key == "" {
			return errors.New("expecting a valid (non-empty) input layer name")
		}

		err := p.SetInput(inputNode.Key, data[ii])
		if err != nil {
			return err
		}
	}

  span, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict", 		
  opentracing.Tags{
    "evaluation_trace_level": p.GetOptions().TraceLevel(),
  })
  defer span.Finish()

  if p.GetOptions().TraceLevel() >= tracer.FRAMEWORK_TRACE {
		// define profiling options
		poptions := map[string]ProfileMode{
			"profile_all":        ProfileAllDisable,
			"profile_symbolic":   ProfileSymbolicOperatorsEnable,
			"profile_imperative": ProfileImperativeOperatorsEnable,
			"profile_memory":     ProfileMemoryEnable,
			"profile_api":        ProfileApiDisable,
			"continuous_dump":    ProfileContinuousDumpDisable,
		}
		if profile, err := NewProfile(poptions, filepath.Join("/tmp", "profile")); err == nil {
			profile.Start()
			defer func() {
				profile.Stop()
				profile.Publish(ctx)
				profile.Delete()
			}()
		}
	}

  err := p.cuptiStart(ctx)
	if err != nil {
		return err
	}

	err = p.Forward()
	if err != nil {
		return err
	}

	p.cuptiClose()

	return nil
}

func (p *Predictor) cuptiStart(ctx context.Context) error {
  opts := p.GetOptions()
	if !opts.UsesGPU() || opts.TraceLevel() < tracer.SYSTEM_LIBRARY_TRACE {
		return nil
  }

  metrics := []string{}
	if opts.GPUMetrics() != "" {
		metrics = strings.Split(opts.GPUMetrics(), ",")
	}

	cu, err := cupti.New(cupti.Context(ctx),
		cupti.SamplingPeriod(0),
		cupti.Metrics(metrics),
	)
	if err != nil {
		return err
	}
	p.cu = cu
  return nil 
}

func (p *Predictor) cuptiClose() {
	if p.cu == nil {
		return
	}
	p.cu.Wait()
	p.cu.Close()
	p.cu = nil
}


// get the shape of output node
// go binding for MXPredGetOutputShape
// param index The index of output node, set to 0 if there is only one output
func (p *Predictor) GetOutputShape(index int) ([]int, error) {
	var (
		shapeData *C.mx_uint = nil
		shapeDim  C.mx_uint  = 0
	)
	success := C.MXPredGetOutputShape(
		p.handle,
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

func (p *Predictor) ReadPredictionOutputAtIndex(ctx context.Context, index int) (gotensor.Tensor, error) {
	node := p.options.OutputNodes()[index]

	if node.Dtype != gotensor.Float32 {
		panic("only supports float32 for now")
  }
  
	shape, err := p.GetOutputShape(index)
	if err != nil {
		return nil, err
	}

	size := prod(shape)
	output := make([]float32, size)
	success := C.MXPredGetOutput(
		p.handle,
		C.mx_uint(index),
		(*C.mx_float)(unsafe.Pointer(&output[0])),
		C.mx_uint(size),
	)
	if success != 0 {
		return nil, GetLastError()
  }
  
	return gotensor.NewDense(node.Dtype, shape, gotensor.WithBacking(output)), nil
}

// get the output of the prediction
// index is the index of the output node, set to 0 assuming there is only one output
func (p *Predictor) ReadPredictionOutputs(ctx context.Context) ([]gotensor.Tensor, error) {
	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_prediction_output")
	defer span.Finish()

	outputNodes := p.options.OutputNodes()
	res := make([]gotensor.Tensor, len(outputNodes))

	for ii := 0; ii < len(outputNodes); ii++ {
		tensor, err := p.ReadPredictionOutputAtIndex(ctx, ii)
		if err != nil {
			return nil, err
		}
		res[ii] = tensor
	}

	return res, nil
}

func (p *Predictor) finalizer() error {
	if p.handle == nil {
		return nil
	}
	success := C.MXPredFree(p.handle)
	if success != 0 {
		return GetLastError()
	}
	return nil
}

// free this predictor's C handle
// go binding for MXPredFree
func (p *Predictor) Close() error {
	if p == nil {
		return nil
	}
	err := p.finalizer()
	p.handle = nil
	return err
}
