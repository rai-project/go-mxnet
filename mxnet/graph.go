package mxnet

import (
	"io/ioutil"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
)

type GraphNode struct {
	Op         string                 `json:"op"`
	Name       string                 `json:"name"`
	Inputs     []interface{}          `json:"inputs"`
	Attributes map[string]interface{} `json:"attrs,omitempty"`
}

type Graph struct {
	Nodes      []GraphNode            `json:"nodes"`
	ArgNodes   []int                  `json:"arg_nodes"`
	NodeRowPtr []int                  `json:"node_row_ptr"`
	Heads      [][]int                `json:"heads"`
	Attributes map[string]interface{} `json:"attrs"`
}

func NewGraph(symbolPath string) (*Graph, error) {
	if !com.IsFile(symbolPath) {
		return nil, errors.Errorf("file path %s not found", symbolPath)
	}
	bts, err := ioutil.ReadFile(symbolPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read %s", symbolPath)
	}
	g := new(Graph)
	if err := json.Unmashal(&g, bts); err != nil {
		return nil, errors.Wrapf(err, "failed to unmashal %s", symbolPath)
	}
	return g, nil
}
