package mxnet

import (
	"github.com/rai-project/config"
	"github.com/rai-project/logger"
	tr "github.com/rai-project/tracer"
	"github.com/sirupsen/logrus"
)

var (
	log    *logrus.Entry
	tracer tr.Tracer
)

func init() {
	config.AfterInit(func() {
		log = logger.New().WithField("pkg", "go-mxnet")
		tracer = tr.MustNew("go-mxnet")
	})
}
