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
  echo "USE_OPENCV=0" >>config.mk

echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH = /usr/local/cuda" >> config.mk
echo "USE_CUDNN=1" >>config.mk
echo "CUDA_ARCH=-gencode=arch=compute_30,code=\"sm_30\" " >> config.mk
echo "CUDA_ARCH+=-gencode=arch=compute_35,code=\"sm_35\" " >> config.mk
echo "CUDA_ARCH+=-gencode=arch=compute_50,code=\"sm_50\" " >> config.mk
echo "CUDA_ARCH+=-gencode=arch=compute_52,code=\"sm_52\" " >> config.mk
echo "CUDA_ARCH+=-gencode=arch=compute_53,code=\"sm_53\" " >> config.mk
echo "CUDA_ARCH+=-gencode=arch=compute_60,code=\"sm_60\" " >> config.mk
echo "CUDA_ARCH+=-gencode=arch=compute_61,code=\"sm_61\" " >> config.mk
echo "CUDA_ARCH+=-gencode=arch=compute_70,code=\"sm_70\" " >> config.mk
echo "ADD_CFLAGS= -ftrack-macro-expansion=0" >>config.mk

make -j"$(nproc)" PREFIX=$MXNET_DIST_DIR

cp -r include $MXNET_DIST_DIR/
cp -r lib $MXNET_DIST_DIR/
