package mxnet

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/Unknwon/com"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/tracer/chrome"
	"golang.org/x/net/context"
)

/*
// go preamble
#cgo pkg-config: mxnet
typedef struct MXCallbackList MXCallbackList;
#include <mxnet/c_api.h>
#include <stdlib.h>
*/
import "C"

type Profile struct {
	*chrome.Trace
	started  bool
	stopped  bool
	dumped   bool
	filename string
}

type ProfileMode int

const (
	ProfileSymbolicOperators = ProfileMode(0)
	ProfileAllOperators      = ProfileMode(1)
)

// go binding for MXSetProfilerConfig()
// param mode kOnlySymbolic: 0, kAllOperator: 1
// param filename output filename
func NewProfile(mode ProfileMode) (*Profile, error) {

	tmpDir := filepath.Join(config.App.TempDir, "mxnet")
	filename, err := tempFile(tmpDir, "profile", ".json")
	if err != nil {
		return nil, errors.Errorf("cannot create temporary file in %v", tmpDir)
	}

	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	success, err := C.MXSetProfilerConfig(C.int(mode), cs)
	if err != nil {
		return nil, err
	}
	if success < 0 {
		return nil, GetLastError()
	}
	return &Profile{
		Trace:    nil,
		filename: filename,
		stopped:  false,
		dumped:   false,
	}, nil
}

// go binding for MXSetProfilerState(1)
func (p *Profile) Start() error {
	success, err := C.MXSetProfilerState(C.int(1))
	if err != nil {
		return err
	}
	if success < 0 {
		return GetLastError()
	}
	p.started = true
	return nil
}

// go binding for MXSetProfilerState(0)
func (p *Profile) Stop() error {
	if !p.started {
		return errors.New("mxnet profile was not started")
	}
	if p.stopped == true {
		return nil
	}
	defer func() {
		p.stopped = true
	}()
	success, err := C.MXSetProfilerState(C.int(0))
	if err != nil {
		return err
	}
	if success < 0 {
		return GetLastError()
	}
	return nil
}

// go binding for MXDumpProfile()
func (p *Profile) Dump() (string, error) {
	if !p.started {
		return "", errors.New("mxnet profile was not started")
	}
	if !p.stopped {
		return "", errors.New("mxnet profile was not stopped")
	}
	if p.dumped == true {
		return "", nil
	}
	defer func() {
		p.dumped = true
	}()
	success, err := C.MXDumpProfile()
	if err != nil {
		return "", err
	}
	if success < 0 {
		return "", GetLastError()
	}
	return p.filename, nil
}

// go binding for MXDumpProfile()
func (p *Profile) String() (string, error) {

	err := p.Read()
	if err != nil {
		return "", err
	}

	bts, err := json.MarshalIndent(p.Trace, "", "\t")
	if err != nil {
		return "", err
	}
	return string(bts), nil
}

func (p *Profile) Read() error {
	if p.Trace != nil {
		return nil
	}

	if !p.started {
		return errors.New("mxnet profile was not started")
	}
	if !p.stopped {
		if err := p.Stop(); err != nil {
			return err
		}
	}
	if !p.dumped {
		if _, err := p.Dump(); err != nil {
			return err
		}
	}

	if !com.IsFile(p.filename) {
		return errors.Errorf("unable to read profile because %v does not exist", p.filename)
	}
	bts, err := ioutil.ReadFile(p.filename)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(bts, p.Trace); err != nil {
		p.Trace = nil
		return err
	}
	return nil
}

func (p *Profile) Delete() error {
	if !com.IsFile(p.filename) {
		return nil
	}
	return os.Remove(p.filename)
}

func (p *Profile) Publish(ctx context.Context, opts ...opentracing.StartSpanOption) (opentracing.Span, context.Context, error) {
	if err := p.Read(); err != nil {
		return nil, nil, err
	}
	return p.Trace.Publish(ctx, "mxnet_profile", opts...)
}
