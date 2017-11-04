 FRAMEWORK_VERSION=v0.11.0

 git clone --single-branch  --branch $FRAMEWORK_VERSION --recursive https://github.com/apache/incubator-mxnet mxnet

 DIST_DIR=$HOME/frameworks/mxnet
 mkdir -p $DIST_DIR

 cd mxnet && \
     cp make/config.mk . && \
     echo "USE_BLAS=openblas" >>config.mk && \
     echo "USE_OPENCV=0" >>config.mk && \
     echo "USE_CPP_PACKAGE=1" >>config.mk && \
     echo "ADD_CFLAGS=-I/usr/include/openblas -Wno-strict-aliasing -Wno-sign-compare -ftrack-macro-expansion=0 -Wno-misleading-indentation -I/usr/local/cuda/include" >>config.mk  && \
     echo "USE_CUDA=1" >>config.mk && \
     echo "USE_CUDNN=1" >>config.mk && \
     echo "ADD_LDFLAGS=-L/usr/local/cuda/lib64" >>config.mk && \
     echo "USE_PROFILER=1" >>config.mk && \
     make PREFIX=$DIST_DIR &&  \
     cp -r include $DIST_DIR/ && \
     cp -r lib $DIST_DIR/

 export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
 export LD_LIBRARY_PATH=$DIST_DIR/lib:$LD_LIBRARY_PATH
