# Sets up python environment with all dependencies

# execture from root directory



# conda remove --name pytorch_cuda --all # remove existing env

# conda env create -f ./pytorch_cuda.yml
# conda activate pytorch_cuda
# pip install nvitop

BASEDIR=$(dirname $0)

curl -o ${BASEDIR}/src/datasets/cifar10/cifar10.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xf ${BASEDIR}/src/datasets/cifar10/cifar10.tar.gz -C ${BASEDIR}/src/datasets/cifar10/

curl -o ${BASEDIR}/src/datasets/cifar100/cifar100.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xf ${BASEDIR}/src/datasets/cifar100/cifar100.tar.gz -C ${BASEDIR}/src/datasets/cifar100/