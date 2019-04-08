package mxnet

import (
	"os"

	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	"github.com/sirupsen/logrus"
)

var (
	log *logrus.Entry
)

func disableFrameworkAutoTuning() {
	os.Setenv("MXNET_EXEC_BULK_EXEC_INFERENCE", "0")
	os.Setenv("MXNET_USE_OPERATOR_TUNING", "0")
	os.Setenv("MXNET_USE_TENSORRT", "0")
	os.Setenv("MXNET_CUDA_ALLOW_TENSOR_CORE", "0")
	os.Setenv("MXNET_CUDNN_AUTOTUNE_DEFAULT", "0")
}

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "go-mxnet")
	})
}
