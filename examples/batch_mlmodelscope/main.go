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
	"github.com/chewxy/math32"
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
	gotensor "gorgonia.org/tensor"
)

// https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/transforms/presets/imagenet.py
// mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
var (
	batchSize   = 8
	model       = "squeezenet1.0"
	shape       = []int{batchSize, 3, 224, 224}
	mean        = []float32{0.485, 0.456, 0.406}
	scale       = []float32{0.229, 0.224, 0.225}
	imgDir, _   = filepath.Abs("../_fixtures")
	imgPath     = filepath.Join(imgDir, "platypus.jpg")
	graph_url   = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/squeezenet1.0/model-symbol.json"
	weights_url = "http://s3.amazonaws.com/store.carml.org/models/mxnet/gluoncv/squeezenet1.0/model-0000.params"
	synset_url  = "http://s3.amazonaws.com/store.carml.org/synsets/imagenet/synset.txt"
)

// convert go Image to 1-dim array
func cvtRGBImageToNCHW1DArray(src image.Image, mean []float32, scale []float32) ([]float32, error) {
	if src == nil {
		return nil, fmt.Errorf("src image nil")
	}

	in := src.(*types.RGBImage)
	height := in.Bounds().Dy()
	width := in.Bounds().Dx()

	out := make([]float32, 3*height*width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			offset := y*in.Stride + x*3
			rgb := in.Pix[offset : offset+3]
			r, g, b := rgb[0], rgb[1], rgb[2]
			out[y*width+x] = (float32(r)/255 - mean[0]) / scale[0]
			out[width*height+y*width+x] = (float32(g)/255 - mean[1]) / scale[1]
			out[2*width*height+y*width+x] = (float32(b)/255 - mean[2]) / scale[2]
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

	height := shape[2]
	width := shape[3]
	channels := shape[1]

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

	imgFloats, err := cvtRGBImageToNCHW1DArray(resized, mean, scale)
	if err != nil {
		panic(err)
	}
	length := len(imgFloats)
	dupImgFloats := make([]float32, length*batchSize)
	for ii := 0; ii < batchSize; ii++ {
		copy(dupImgFloats[ii*length:(ii+1)*length], imgFloats)
	}
	input := []*gotensor.Dense{gotensor.New(
		gotensor.Of(gotensor.Float32),
		gotensor.WithShape(batchSize, height, width, channels),
		gotensor.WithBacking(dupImgFloats),
	),
	}
	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	ctx := context.Background()

	span, ctx := tracer.StartSpanFromContext(ctx, tracer.FULL_TRACE, "mxnet_batch")
	defer span.Finish()

	in := options.Node{
		Key:   "data",
		Shape: shape,
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
				Dtype: gotensor.Float32,
			},
		}),
	)
	if err != nil {
		panic(fmt.Sprintf("%v", err))
	}
	defer predictor.Close()

	err = predictor.Predict(ctx, input)
	if err != nil {
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
		soutputs := output[ii*featuresLen : (ii+1)*featuresLen]
		for jj := 0; jj < featuresLen; jj++ {
			rprobs[jj] = feature.New(
				feature.ClassificationIndex(int32(jj)),
				feature.ClassificationLabel(labels[jj]),
				feature.Probability(soutputs[jj]),
			)
		}
		nprobs := dlframework.Features(rprobs).ProbabilitiesApplySoftmaxFloat32()
		sort.Sort(nprobs)
		features[ii] = nprobs
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

// SoftMax performs softmax on the input. Specifically this is used:
//		e^(a[i]) / sum((e^(a[i])))
func softmax(ts []float32) []float32 {
	res := make([]float32, len(ts))
	accum := float32(0.0)
	for ii, t := range ts {
		res[ii] = math32.Exp(t)
		accum += res[ii]
	}
	for ii, r := range res {
		res[ii] = r / accum
	}
	return res
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
