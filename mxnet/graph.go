package mxnet

import (
	"encoding/json"
	"io/ioutil"
	"sort"

	"github.com/Unknwon/com"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
	"gonum.org/v1/gonum/graph/topo"
)

type GraphNode struct {
	id         int64                  `json:"-"`
	Op         string                 `json:"op"`
	Name       string                 `json:"name"`
	Inputs     [][]int64              `json:"inputs"`
	Attributes map[string]interface{} `json:"attrs,omitempty"`
}

type Graph struct {
	id         int64                  `json:"-"`
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
	if err := json.Unmarshal(bts, g); err != nil {
		return nil, errors.Wrapf(err, "failed to unmashal %s", symbolPath)
	}
	return g, nil
}

func (nd GraphNode) ID() int64 {
	return nd.id
}

func (g Graph) ID() int64 {
	return g.id
}

func (g *Graph) TopologicallySortedNodes() ([]GraphNode, error) {

	graphIdxs := map[int64]string{}
	graphIds := map[string]int64{}

	grph := simple.NewDirectedGraph()
	for ii, nd := range g.Nodes {
		nd.id = grph.NewNode().ID()
		grph.AddNode(nd)
		graphIdxs[int64(ii)] = nd.Name
		graphIds[nd.Name] = nd.ID()
	}
	for _, ind := range g.Nodes {
		outNodeId, ok := graphIds[ind.Name]
		if !ok {
			continue
		}
		outNode := grph.Node(outNodeId)
		for _, ond := range ind.Inputs {
			if len(ond) != 2 {
				continue
			}
			name, ok := graphIdxs[ond[0]]
			if !ok {
				continue
			}
			outNodeId, ok := graphIds[name]
			if !ok {
				continue
			}
			inNode := grph.Node(outNodeId)
			grph.SetEdge(grph.NewEdge(inNode, outNode))
		}
	}

	nds, err := topo.SortStabilized(grph, sortById)
	if err != nil {
		return nil, errors.Wrap(err, "failed to topologically sort graph")
	}

	res := []GraphNode{}
	for _, g := range nds {
		res = append(res, g.(GraphNode))
	}
	return res, nil
}

type byID []graph.Node

func (n byID) Len() int           { return len(n) }
func (n byID) Less(i, j int) bool { return n[i].ID() < n[j].ID() }
func (n byID) Swap(i, j int)      { n[i], n[j] = n[j], n[i] }

func sortById(nodes []graph.Node) {
	sort.Sort(byID(nodes))
}
