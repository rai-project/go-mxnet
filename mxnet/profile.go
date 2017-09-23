package mxnet

/*
// go preamble
#cgo pkg-config: mxnet
typedef struct MXCallbackList MXCallbackList;
#include <mxnet/c_api.h>
#include <stdlib.h>
*/
import "C"
import "unsafe"

// go binding for MXSetProfilerConfig()
// param mode kOnlySymbolic: 0, kAllOperator: 1
// param filename output filename
func ProfilerConfig(mode int, filename string) error {
	cs := C.CString(filename)
	defer C.free(unsafe.Pointer(cs))
	success, err := C.MXSetProfilerConfig(C.int(mode), cs)
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

// go binding for MXDumpProfile()
func ProfilerDump() error {
	success, err := C.MXDumpProfile()
	if err != nil {
		return err
	} else if success < 0 {
		return GetLastError()
	}
	return nil
}
