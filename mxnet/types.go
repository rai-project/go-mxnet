package mxnet

type device struct {
	id         int        // device id
	deviceType DeviceType // device type
}

type inputNode struct {
	key   string   // name
	shape []uint32 // shape of ndarray
}

type DeviceType int

const (
	CPU_DEVICE DeviceType = iota + 1 // cpu device type
	GPU_Device                       // gpu device type
)
