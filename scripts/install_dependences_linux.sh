GCC_VERSION=5

apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && apt-get install -y --no-install-recommends \
        gcc-${GCC_VERSION} \
        g++-${GCC_VERSION}  \
        libvips          \
        libjpeg-turbo8-dev \
        libturbojpeg \
        libvips-dev \
        libvips42 \
        build-essential \
        pkg-config \
        git \
        libopenblas-dev \
        libopenblas-base \
        python-dev \
        libcurl4-openssl-dev \
        libgtest-dev \
        cmake \
        curl \
        wget \
        unzip \
        apt-transport-https \
        ca-certificates \
    && \
    #rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${GCC_VERSION} 60 --slave /usr/bin/g++ g++ /usr/bin/g++-${GCC_VERSION} && \
    update-alternatives --config gcc && \
    gcc --version

# add powerai repo
curl -fsSL https://public.dhe.ibm.com/software/server/POWER/Linux/mldl/ubuntu/mldl-repo-network_3.4.0_ppc64el.deb -O && \
	dpkg -i mldl-repo-network_3.4.0_ppc64el.deb && \
	apt-get update

# caffe specific
apt-get update && apt-get install -y --no-install-recommends \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler

# caffe2 specific
apt-get update && apt-get install -y --no-install-recommends \
        libgoogle-glog-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libgflags-dev \
        libiomp-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopenmpi-dev \
        libsnappy-dev \
        openmpi-bin \
        openmpi-doc \
        libeigen3-dev
