#!/bin/bash
MPWD=`pwd`

checkRet() {
	ret=$1
	if [[ $ret -eq 0 ]];then
		return
	else
		exit 1
	fi	
}

cd $MPWD/travis && git clone https://github.com/dmlc/mxnet.git --recursive
cd $MPWD/travis/mxnet && make -j4 USE_BLAS=openblas USE_OPENCV=1
#TODO fix mxnet building in travis
#checkRet $?

go get github.com/anthonynsimon/bild

sed -i "/prefix=/c prefix=$MPWD" $MPWD/travis/mxnet.pc
sudo cp $MPWD/travis/mxnet.pc /usr/lib/pkgconfig/
pkg-config --libs --cflags mxnet
checkRet $?

exit 0
