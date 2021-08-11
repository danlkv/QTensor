from centos:latest

run mkdir /myapp



run yum update -y
run yum groupinstall -y "Development Tools"
run yum install -y gcc
run yum install -y gcc-c++
run yum install -y wget

copy ./app/pi.c /myapp/
copy ./app/Makefile /myapp/
copy ./app/build-mpi.sh /myapp/

run myapp/build-mpi.sh
env PATH=$PATH:$PWD/mpich-3.2.1/install/bin
env LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/mpich-3.2.1/install/lib
run cd /myapp && make

entrypoint ["/myapp/pi"]
