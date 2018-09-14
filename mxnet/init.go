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

func init() {
	os.Setenv("MXNET_EXEC_BULK_EXEC_INFERENCE","0")
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "go-mxnet")
	})
}
