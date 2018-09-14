package mxnet

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"context"

	"github.com/Unknwon/com"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"github.com/rai-project/config"
	"github.com/rai-project/tracer/chrome"
)

/*
// go preamble
typedef struct MXCallbackList MXCallbackList;
#include <mxnet/c_api.h>
#include <stdlib.h>

*/
import "C"

var initTime = time.Now()

type Profile struct {
	Trace          *chrome.Trace
	startTime      time.Time
	lastPauseTime  time.Time
	lastResumeTime time.Time
	endTime        time.Time
	started        bool
	paused         bool
	stopped        bool
	dumped         bool
	filename       string
}

// profile type
type ProfileMode string

// profile options
const (
	ProfileAllDisable                 = ProfileMode(0)
	ProfileSymbolicOperatorsDisable   = ProfileMode(0)
	ProfileImperativeOperatorsDisable = ProfileMode(0)
	ProfileMemoryDisable              = ProfileMode(0)
	ProfileApiDisable                 = ProfileMode(0)
	ProfileContiguousDumpDisable      = ProfileMode(0)
	ProfileAllEnable                  = ProfileMode(1)
	ProfileSymbolicOperatorsEnable    = ProfileMode(1)
	ProfileImperativeOperatorsEnable  = ProfileMode(1)
	ProfileMemoryEnable               = ProfileMode(1)
	ProfileApiEnable                  = ProfileMode(1)
	ProfileContiguousDumpEnable       = ProfileMode(1)
	ProfileDumpPeriod                 = ProfileMode(1)
)

// go binding for MXSetProfilerConfig()
// param profile_options map of profiling options
// param tmpDir output filepath
func NewProfile(profileOptions map[string]ProfileMode, tmpDir string) (*Profile, error) {
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
	keys := [8]string{"filename", "profile_all", "profile_symbolic", "profile_imperative", "profile_memory", "profile_api", "contiguous_dump", "dump_period"}

	// convert go data structures into c data structures
	ckeys := C.malloc(C.size_t(8) * C.size_t(unsafe.Sizeof(uintptr(0))))
	a := (*[1<<30 - 1]*C.char)(ckeys)
	cvals := C.malloc(C.size_t(8) * C.size_t(unsafe.Sizeof(uintptr(0))))
	b := (*[1<<30 - 1]*C.char)(cvals)
	a[0] = C.CString("filename")
	b[0] = cs
	for i := 1; i < 8; i++ {
		a[i] = C.CString(keys[i])
		b[i] = C.CString(string(profileOptions[keys[i]]))
	}

	success, err := C.MXSetProfilerConfig(C.int(len(keys)), (**C.char)(ckeys), (**C.char)(cvals))
	if err != nil {
		return nil, err
	}
	if success != 0 {
		return nil, GetLastError()
	}

	// free C pointers
	C.free(unsafe.Pointer(ckeys))
	C.free(unsafe.Pointer(cvals))

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

// go binding for MXProfilePause(1)
func (p *Profile) Pause() error {
	if !p.started {
		return errors.New("mxnet profile was not started")
	}
	if p.stopped == true || p.paused == true {
		return nil
	}
	defer func() {
		p.paused = true
	}()
	p.lastPauseTime = time.Now()
	success, err := C.MXProfilePause(C.int(1))
	if err != nil {
		return err
	}
	if success != 0 {
		return GetLastError()
	}

	return nil
}

// go binding for MXProfilePause(0)
func (p *Profile) Resume() error {
	if !p.started {
		return errors.New("mxnet profile was not started")
	}
	if p.stopped == true || p.paused == false {
		return nil
	}
	defer func() {
		p.paused = false
	}()
	p.lastResumeTime = time.Now()
	success, err := C.MXProfilePause(C.int(0))
	if err != nil {
		return err
	}
	if success != 0 {
		return GetLastError()
	}

	return nil
}

// go binding for MXDumpProfile()
func (p *Profile) Dump(finished bool) (string, error) {
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
	var fin int
	if finished {
		fin = 1
	}
	success, err := C.MXDumpProfile(C.int(fin))
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
		if _, err := p.Dump(true); err != nil {
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

func (p *Profile) Publish(ctx context.Context, opts ...opentracing.StartSpanOption) error {
	if err := p.Read(); err != nil {
		return err
	}
	if pth, ok := ctx.Value("graph_path").(string); ok {
		p.addNodeMetadata(pth)
	}
	return p.Trace.Publish(ctx, opts...)
}

func (p *Profile) addNodeMetadata(pth string) {
	grph, err := NewGraph(pth)
	if err != nil {
		return
	}
	for event := range p.Trace.TraceEvents {

	}
}
