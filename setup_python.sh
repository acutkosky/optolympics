#install pyenv and virtualenv to manage python versions.
#sets default to python3.6.3

sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev
curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
pyenv update
echo eval "$(pyenv init -)" >> ~/.bash_profile
source ~/.bash_profile

pyenv install -v 3.6.3
pyenv virtualenv 3.6.3 py3.6.3env
pyenv global py3.6.3env

pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
pip3 install torchvision