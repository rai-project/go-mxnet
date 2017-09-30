package main

import (
	"bufio"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"

	"github.com/k0kubun/pp"

	"github.com/anthonynsimon/bild/imgio"
	"github.com/anthonynsimon/bild/transform"
	"github.com/rai-project/config"
	"github.com/rai-project/downloadmanager"
	"github.com/rai-project/go-mxnet-predictor/mxnet"
	"github.com/rai-project/go-mxnet-predictor/utils"
)

var (
	graph_url    = "http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.0-symbol.json"
	weights_url  = "http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.0-0000.params"
	features_url = "http://data.dmlc.ml/mxnet/models/imagenet/synset.txt"
)

func main() {
	dir, _ := filepath.Abs("../tmp")
	graph := filepath.Join(dir, "squeezenet_v1.0-symbol.json")
	weights := filepath.Join(dir, "squeezenet_v1.0-0000.params")
	features := filepath.Join(dir, "synset.txt")

	if _, err := downloadmanager.DownloadInto(graph_url, dir); err != nil {
		os.Exit(-1)
	}

	if _, err := downloadmanager.DownloadInto(weights_url, dir); err != nil {
		os.Exit(-1)
	}
	if _, err := downloadmanager.DownloadInto(features_url, dir); err != nil {
		os.Exit(-1)
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

	// load test image for predction
	img, err := imgio.Open("../_fixtures/platypus.jpg")
	if err != nil {
		panic(err)
	}
	// preprocess
	resized := transform.Resize(img, 224, 224, transform.Linear)
	res, err := utils.CvtImageTo1DArray(resized)
	if err != nil {
		panic(err)
	}

	// create predictor
	p, err := mxnet.CreatePredictor(
		mxnet.Symbol(symbol),
		mxnet.Weights(params),
		mxnet.InputNode("data", []uint32{3, 224, 224}),
		mxnet.Device(2, 1),
	)
	if err != nil {
		pp.Println(mxnet.GetLastError())
		panic(err)
	}
	defer p.Free()

	// set input
	if err := p.SetInput("data", res); err != nil {
		panic(err)
	}

	// do predict
	if err := p.Forward(); err != nil {
		panic(err)
	}

	// get predict result
	data, err := p.GetOutput(0)
	if err != nil {
		panic(err)
	}
	idxs := make([]int, len(data))
	for i := range data {
		idxs[i] = i
	}
	as := utils.ArgSort{Args: data, Idxs: idxs}
	sort.Sort(as)

	var labels []string
	f, err := os.Open(features)
	if err != nil {
		os.Exit(-1)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		labels = append(labels, line)
	}

	pp.Println(as.Args[0])
	pp.Println(labels[as.Idxs[0]])

	// os.RemoveAll(dir)
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
