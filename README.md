# Point Cloud Network: Code, Paper, Materials

This repository holds all of the supplemental materials for the [_Point Cloud Network: An Order of Magnitude Improvement in Linear Layer Parameter Count_](https://gitlab.com/cHetterich/pcn-paper-and-materials/-/raw/main/proposal.pdf?ref_type=heads&inline=false) paper.

## Preliminary Tests

All preliminary testing code can be found in the `preliminary_testing` directory. The general structure is a series of experiments, with a dedicated folder for each experiment. With the exception of the last experiment, each experiment contains a `test.ipynb` and `README.md`. **Datasets not included**.

Experiments 1-3 are most relevant to design and regularization choices seen in the paper. Experiments 4 & 4.5 are attempts at building a CUDA kernel for the PCN forward & backward operations. Experiments 5-7 are largely irrelevant and just precursory to the paper.

## Reproducing Trials

Here is a brief guide on how you can reproduce the results from this paper renting a GPU from [vast.ai](https://vast.ai/).

1. **vast.ai**

Create a vast.ai account, add funds to your account ($20 should be sufficient for renting a 4090 RTX and running through the experiments), and add your SSH key to your account.

##### [ [vast.ai quickstart](https://vast.ai/docs/overview/quickstart) ]

When you have a functional account, rent a GPU instance of your choosing with the [pytorch docker image](https://hub.docker.com/r/pytorch/pytorch/).

2. **Jupyter**

Connect to your instance's Jupyter Notebook, and use the following code blocks to setup and run a given trial.

```
# Setup python, code, & datasets

!git clone https://gitlab.com/cHetterich/pcn-paper-and-materials.git
!sh ./pcn-paper-and-materials/setup.sh
!pip install tensorboard nvitop
```

##### **(** the command `nvitop -m full` can be used on the server to monitor GPU usage **)**

```
# Launch tensorboard

%load_ext tensorboard
%tensorboard --logdir log_dir --reload_interval 1 --port 6006
```

```
# Run trial

!python pcn-paper-and-materials/src/train_alexnet.py --epochs 3500 --batch_size 1024 --dataset cifar100 --out_dir model_out  --num_workers 0
```

3. **Tensorboard**

To view the tensorboard, run the following, filling in `<port>` and `<instance-ip>`, in order to connect to the instance with an ssh tunnel

```
ssh -p <port> root@<instance-ip> -L 8080:localhost:8080 -L 6006:localhost:6006
```

You should now be able to see the training run at `localhost:6006` in your browser.

4. **Results**

After the run completes, models will be saved in the `./model_out/<dataset-name>/` directory, relative to the Jupyter Notebook's location.

## License

MIT License

Copyright (c) 2023 Charles Hetterich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
