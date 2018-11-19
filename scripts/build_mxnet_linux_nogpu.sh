#!/bin/sh

FRAMEWORK_VERSION=1.3.0
MXNET_SRC_DIR=$HOME/code/mxnet
MXNET_DIST_DIR=/opt/mxnet

if [ ! -d "$MXNET_SRC_DIR" ]; then
  git clone --single-branch --depth 1 --branch $FRAMEWORK_VERSION --recursive https://github.com/apache/incubator-mxnet $MXNET_SRC_DIR
fi

if [ ! -d "$MXNET_DIST_DIR" ]; then
  mkdir -p $MXNET_DIST_DIR
fi

cd $MXNET_SRC_DIR && cp make/config.mk . && \
  echo "USE_BLAS=openblas" >>config.mk && \
  echo "USE_CPP_PACKAGE=1" >>config.mk && \
  echo "ADD_CFLAGS=-I/usr/include/openblas -Wno-strict-aliasing -Wno-sign-compare  -Wno-misleading-indentation -I/usr/local/cuda/include" >>config.mk  && \
  echo "ADD_LDFLAGS=-L/usr/local/cuda/lib64" >>config.mk && \
  echo "USE_PROFILER=1" >>config.mk && \
  echo "USE_OPENCV=0" >>config.mk \
  echo "USE_CUDA=0" >>config.mk && \
  echo "USE_CUDNN=0" >>config.mk

make -j"$(nproc)" PREFIX=$MXNET_DIST_DIR

cp -r include $MXNET_DIST_DIR/
cp -r lib $MXNET_DIST_DIR/
