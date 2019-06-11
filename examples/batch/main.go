package main

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework"
	"github.com/rai-project/dlframework/framework/feature"
	"github.com/rai-project/dlframework/framework/options"
	"github.com/rai-project/downloadmanager"
	cupti "github.com/rai-project/go-cupti"
	"github.com/rai-project/go-mxnet/mxnet"
	raiimage "github.com/rai-project/image"
	"github.com/rai-project/image/types"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	_ "github.com/rai-project/tracer/all"
	"gorgonia.org/tensor"
	gotensor "gorgonia.org/tensor"
)

var (
	batchSize   = 1
	model       = "alexnet"
	graph_url   = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/alexnet/model-symbol.json"
	weights_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/alexnet/model-0000.params"
	synset_url  = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt"
)

// convert go Image to 1-dim array
func cvtRGBImageToNCHW1DArray(src image.Image, mean []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	in := src.(*types.RGBImage)
	height := in.Bounds().Dy()
	width := in.Bounds().Dx()
	scale := []float32{0.229, 0.224, 0.225}

	out := make([]float32, 3*height*width)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := y*in.Stride + x*3
			rgb := in.Pix[offset : offset+3]
			r, g, b := rgb[0], rgb[1], rgb[2]
			out[y*width+x] = (float32(r)/255 - mean[0]) / scale[0]
			out[width*height+y*width+x] = (float32(g)/255 - mean[1]) / scale[1]
			out[2*width*height+y*width+x] = (float32(b)/255 - mean[2]) / scale[2]

			// out[offset+0] = (float32(r)/255 - mean[0]) / scale[0]
			// out[offset+1] = (float32(g)/255 - mean[1]) / scale[1]
			// out[offset+2] = (float32(b)/255 - mean[2]) / scale[2]
		}
	}
	return out, nil
}

func main() {
	defer tracer.Close()

	dir, _ := filepath.Abs("../tmp")
	dir = filepath.Join(dir, model)
	graph := filepath.Join(dir, "model-symbol.json")
	weights := filepath.Join(dir, "model-0000.params")
	synset := filepath.Join(dir, "synset.txt")

	if !com.IsFile(graph) {
		if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
			panic(err)
		}
	}
	if !com.IsFile(weights) {
		if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
			panic(err)
		}
	}
	if !com.IsFile(synset) {
		if _, err := downloadmanager.DownloadInto(synset_url, dir); err != nil {
			panic(err)
		}
	}

	// load model
	symbol, err := ioutil.ReadFile(graph)
	if err != nil {
		panic(err)
	}
	params, err := ioutil.ReadFile(weights)
	if err != nil {
		panic(err)
	}

	height := 224
	width := 224
	channels := 3

	imgDir, _ := filepath.Abs("../_fixtures")
	imgPath := filepath.Join(imgDir, "cheeseburger.jpg")
	r, err := os.Open(imgPath)
	if err != nil {
		panic(err)
	}

	var imgOpts []raiimage.Option
	imgOpts = append(imgOpts, raiimage.Mode(types.RGBMode))
	img, err := raiimage.Read(r, imgOpts...)
	if err != nil {
		panic(err)
	}

	imgOpts = append(imgOpts, raiimage.Resized(height, width))
	imgOpts = append(imgOpts, raiimage.ResizeAlgorithm(types.ResizeAlgorithmLinear))
	resized, err := raiimage.Resize(img, imgOpts...)
	if err != nil {
		panic(err)
	}

	input := make([]gotensor.Tensor, batchSize)
	imgFloats, err := cvtRGBImageToNCHW1DArray(resized, []float32{0.485, 0.456, 0.406} /* []float32{123, 117, 104} */)
	if err != nil {
		panic(err)
	}

	pp.Println(resized.(*types.RGBImage).Pix[:4])

	pp.Println(imgFloats[:4])

	for ii := 0; ii < batchSize; ii++ {
		input[ii] = gotensor.New(
			gotensor.Of(tensor.Float32),
			gotensor.WithShape(height, width, channels),
			gotensor.WithBacking(imgFloats),
		)
	}

	pp.Println(input[0].At(0, 0, 0))

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	ctx := context.Background()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "mxnet_batch")
	defer span.Finish()

	in := options.Node{
		Key:   "data",
		Shape: []int{1, 3, 224, 224},
	}

	predictor, err := mxnet.New(
		ctx,
		options.WithOptions(options.New()),
		options.Device(device, 0),
		options.Graph(symbol),
		options.Weights(params),
		options.BatchSize(batchSize),
		options.InputNodes([]options.Node{in}),
		options.OutputNodes([]options.Node{
			options.Node{
				Dtype: tensor.Float32,
			},
		}),
	)
	if err != nil {
		panic(fmt.Sprintf("%v", err))
	}
	defer predictor.Close()

	err = predictor.Predict(ctx, input)
	if err != nil {
		pp.Println("fine")

		panic(err)
	}

	enableCupti := false
	var cu *cupti.CUPTI
	if enableCupti && nvidiasmi.HasGPU {
		cu, err = cupti.New(cupti.Context(ctx))
		if err != nil {
			panic(err)
		}
	}

	// define profiling options
	profileOptions := map[string]mxnet.ProfileMode{
		"profile_all":        mxnet.ProfileAllDisable,
		"profile_symbolic":   mxnet.ProfileSymbolicOperatorsEnable,
		"profile_imperative": mxnet.ProfileImperativeOperatorsDisable,
		"profile_memory":     mxnet.ProfileMemoryDisable,
		"profile_api":        mxnet.ProfileApiDisable,
		"continuous_dump":    mxnet.ProfileContinuousDumpDisable,
	}

	profile, err := mxnet.NewProfile(profileOptions, "")
	if err != nil {

		panic(err)
	}
	profile.Start()

	err = predictor.Predict(ctx, input)
	if err != nil {
		panic(err)
	}

	profile.Stop()
	profile.Publish(ctx)
	profile.Delete()

	if enableCupti && nvidiasmi.HasGPU {
		cu.Wait()
		cu.Close()
	}

	outputs, err := predictor.ReadPredictionOutputs(ctx)
	if err != nil {
		panic(err)
	}

	if len(outputs) != 1 {
		panic(errors.Errorf("invalid output length. got outputs of length %v", len(outputs)))
	}

	output := outputs[0].Data().([]float32)

	var labels []string
	f, err := os.Open(synset)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	features := make([]dlframework.Features, batchSize)
	featuresLen := len(output) / batchSize

	for ii := 0; ii < batchSize; ii++ {
		rprobs := make([]*dlframework.Feature, featuresLen)
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(output[ii*featuresLen+jj]),
			)
		}
		sort.Sort(dlframework.Features(rprobs))
		features[ii] = rprobs
	}

	if true {
		for i := 0; i < 1; i++ {
			results := features[i]
			top1 := results[0]
			pp.Println(top1.Probability)
			pp.Println(top1.GetClassification().GetLabel())
		}
	} else {
		_ = features
	}
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
