FRAMEWORK_VERSION=v0.11.0

SRC_DIR=$HOME/code/mxnet
DIST_DIR=$HOME/frameworks/mxnet

mkdir -p $DIST_DIR

git clone --single-branch --depth 1 --branch $FRAMEWORK_VERSION --recursive https://github.com/apache/incubator-mxnet $SRC_DIR

cd $SRC_DIR && \
  cp make/config.mk . && \
  echo "USE_BLAS=openblas" >>config.mk && \
  echo "USE_CPP_PACKAGE=1" >>config.mk && \
  echo "ADD_CFLAGS=-I/usr/include/openblas -Wno-strict-aliasing -Wno-sign-compare  -Wno-misleading-indentation -I/usr/local/cuda/include" >>config.mk  && \
  echo "ADD_LDFLAGS=-L/usr/local/cuda/lib64" >>config.mk && \
  echo "USE_PROFILER=1" >>config.mk && \
  echo "USE_OPENCV=0" >>config.mk

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  echo "USE_CUDA=1" >>config.mk
  echo "USE_CUDNN=1" >>config.mk
  echo "ADD_CFLAGS= -ftrack-macro-expansion=0" >>config.mk
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "USE_OPENMP=0" >> config.mk
  echo "USE_BLAS=apple" >> config.mk
fi

make -j8 PREFIX=$DIST_DIR &&  \
  cp -r include $DIST_DIR/ && \
  cp -r lib $DIST_DIR/

 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
 export LD_LIBRARY_PATH=$DIST_DIR/lib:$LD_LIBRARY_PATH
