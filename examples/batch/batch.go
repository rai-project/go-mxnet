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
	batch        = 10
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

	// create predictor
	p, err := mxnet.CreatePredictor(symbol,
		params,
		mxnet.Device{mxnet.CPU_DEVICE, 0},
		[]mxnet.InputNode{{Key: "data", Shape: []uint32{uint32(batch), 3, 224, 224}}},
	)
	if err != nil {
		panic(err)
	}
	defer p.Free()

	var input []float32
	cnt := 0

	dir, _ = filepath.Abs("../_fixtures")
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if path == dir || filepath.Ext(path) != ".jpg" || cnt >= batch {
			return nil
		}

		img, err := imgio.Open(path)
		if err != nil {
			return err
		}
		resized := transform.Resize(img, 224, 224, transform.Linear)
		res, err := utils.CvtImageTo1DArray(resized)
		if err != nil {
			panic(err)
		}
		input = append(input, res...)
		cnt++

		return nil
	})
	if err != nil {
		panic(err)
	}

	padding := make([]float32, (batch-cnt)*3*224*224)
	input = append(input, padding...)

	// set input
	if err := p.SetInput("data", input); err != nil {
		panic(err)
	}

	profiler, err := mxnet.NewProfile(mxnet.ProfileAllOperators)
	if err != nil {
		panic(err)
	}

	profiler.Start()

	// do predict
	if err := p.Forward(); err != nil {
		panic(err)
	}

	profiler.Stop()

	// get predict result
	output, err := p.GetOutput(0)
	if err != nil {
		panic(err)
	}
	len := len(output) / batch

	for i := 0; i < cnt; i++ {

		idxs := make([]int, len)
		for j := 0; j < len; j++ {
			idxs[j] = j
		}
		as := utils.ArgSort{Args: output[i*len : (i+1)*len], Idxs: idxs}
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
	}

	// dump profiling at the end
	filename, err := profiler.Dump()
	if err != nil {
		panic(err)
	}
	pp.Println(filename)

	// os.RemoveAll(dir)
}

func init() {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)
}
