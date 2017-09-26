package mxnet

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
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

var initTime time.Time

type Profile struct {
	Trace     *chrome.Trace
	startTime time.Time
	endTime   time.Time
	started   bool
	stopped   bool
	dumped    bool
	filename  string
}

type ProfileMode int

const (
	ProfileSymbolicOperators = ProfileMode(0)
	ProfileAllOperators      = ProfileMode(1)
)

// go binding for MXSetProfilerConfig()
// param mode kOnlySymbolic: 0, kAllOperator: 1
// param filename output filename
func NewProfile(mode ProfileMode, tmpDir string) (*Profile, error) {
	if tmpDir == "" {
		tmpDir = filepath.Join(config.App.TempDir, "mxnet", "profile")
	}
	if !com.IsDir(tmpDir) {
		os.MkdirAll(tmpDir, os.FileMode(0755))
	}
	filename, err := tempFile(tmpDir, "profile-", ".json")
	if err != nil {
		return nil, errors.Errorf("cannot create temporary file in %v", tmpDir)
	}

	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	success, err := C.MXSetProfilerConfig(C.int(mode), cs)
	if err != nil {
		return nil, err
	}
	if success != 0 {
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
	if success != 0 {
		return GetLastError()
	}
	p.startTime = time.Now()
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
	p.endTime = time.Now()
	success, err := C.MXSetProfilerState(C.int(0))
	if err != nil {
		return err
	}
	if success != 0 {
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
	if success != 0 {
		return "", GetLastError()
	}

	return p.filename, nil
}

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
	p.Trace = new(chrome.Trace)
	if err := json.Unmarshal(bts, p.Trace); err != nil {
		p.Trace = nil
		return errors.Wrapf(err, "failed to unmarshal %v", p.filename)
	}
	p.Trace.TimeUnit = "us"
	p.Trace.StartTime = p.startTime
	p.Trace.EndTime = p.endTime

	p.process()

	return nil
}

func (p *Profile) process() {

	timeUnit := time.Microsecond

	start := p.startTime

	minTime := int64(0)
	events := []chrome.TraceEvent{}
	for _, event := range p.Trace.TraceEvents {
		if event.EventType != "B" && event.EventType != "E" {
			continue
		}
		t := initTime.Add(time.Duration(event.Timestamp) * timeUnit)
		if start.After(t) {
			continue
		}
		events = append(events, event)
		if event.EventType != "B" {
			continue
		}
		if minTime != 0 && minTime < event.Timestamp {
			continue
		}
		minTime = event.Timestamp
	}

	for ii, event := range events {
		events[ii].Name = strings.Trim(strings.Trim(event.Name, "["), "]")
		events[ii].Time = start.Add(time.Duration(event.Timestamp-minTime) * timeUnit)
	}

	p.Trace.TraceEvents = events
}

func (p *Profile) Delete() error {
	if !com.IsFile(p.filename) {
		return nil
	}

	return os.Remove(p.filename)
}

func (p *Profile) Publish(ctx context.Context, operationName string, opts ...opentracing.StartSpanOption) error {
	if err := p.Read(); err != nil {
		return err
	}
	return p.Trace.Publish(ctx, tracer, opts...)
}

func init() {
	initTime = time.Now()
}
