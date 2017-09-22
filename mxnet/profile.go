package mxnet

/*
// go preamble
#cgo pkg-config: mxnet
#include <mxnet/c_api.h>
#include <stdlib.h>
*/
import "C"

// go binding for MXSetProfilerConfig()
// param mode kOnlySymbolic: 0, kAllOperator: 1
// param filename output filename
func ProfilerConfig(mode int, filename string) error {
	success, err := C.MXSetProfilerConfig(C.int(mode), C.CString(filename))
	if err != nil {
		return err
	} else if success < 0 {
		return GetLastError()
	}
	return nil
}

// go binding for MXSetProfilerState(1)
func ProfilerStart() error {
	success, err := C.MXSetProfilerState(C.int(1))
	if err != nil {
		return err
	} else if success < 0 {
		return GetLastError()
	}
	return nil
}

// go binding for MXSetProfilerState(0)
func ProfilerStop() error {
	success, err := C.MXSetProfilerState(C.int(0))
	if err != nil {
		return err
	} else if success < 0 {
		return GetLastError()
	}
	return nil
}
