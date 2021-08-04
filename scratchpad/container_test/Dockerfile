from centos:latest

run mkdir /myapp


copy ./app/pi.c /myapp/
copy ./app/Makefile /myapp/


run yum update -y
run yum groupinstall -y "Development Tools"
run yum install -y gcc
run yum install -y gcc-c++
run yum install -y wget
run cd /myapp
run make

entrypoint ["/myapp/pi"]
