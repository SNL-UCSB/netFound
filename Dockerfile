FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev  \
    wget \
    graphviz \
    parallel \
    nano vim make cmake g++ \
    libpcap-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN wget https://github.com/seladb/PcapPlusPlus/releases/download/v24.09/pcapplusplus-24.09-ubuntu-22.04-gcc-11.4.0-x86_64.tar.gz && \
    tar -xvf pcapplusplus-24.09-ubuntu-22.04-gcc-11.4.0-x86_64.tar.gz && \
    rm pcapplusplus-24.09-ubuntu-22.04-gcc-11.4.0-x86_64.tar.gz && \
    mv pcapplusplus-24.09-ubuntu-22.04-gcc-11.4.0-x86_64 /usr/local/ && \
    ln -s /usr/local/pcapplusplus-24.09-ubuntu-22.04-gcc-11.4.0-x86_64 /usr/local/pcapplusplus

ENV PATH="/usr/local/pcapplusplus/bin:$PATH"

COPY . .
RUN find . -type f -name "*.sh" -exec chmod +x {} \;

CMD ["bash"]
