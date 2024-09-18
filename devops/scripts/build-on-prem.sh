docker build -f ./devops/dockerfile/server-agent/Dockerfile-Release-OnPrem -t fedml/fedml-server-agent:release-on-prem .
docker push fedml/fedml-server-agent:release-on-prem
docker build -f ./devops/dockerfile/device-image/Dockerfile-Release-OnPrem -t fedml/fedml-device-image:local .
docker push  fedml/fedml-device-image:local
