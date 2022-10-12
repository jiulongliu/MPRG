## This repository contains the code for the paper:

Misspecified Phase Retrieval with Generative Priors

-------------------------------------------------------------------------------------

## Dependencies

* Python 3.6

* Tensorflow 1.5.0

* Scipy 1.1.0

*  PyPNG

## Running the code

-------------------------------------------------------------------------------------

We provide the guideline to run our method MPRG and to compare it with several methods on the MNIST and CelebA datasets. 

(1) Run experiments on MNIST dataset 

(1-1) MNIST image recovery from measurement abs(Ax)+eta
python mnist_main_mpr.py  --nonlinear-model 'abs(Ax)+eta'  --num-outer-measurement-ls 200 300 400 500 600 700 800   --max-update-iter 120  --method-ls MPRS PPower MPRG_Step2 APPGD MPRG --noise-std-ls  0.0 0.01 0.1 0.5 1.0  

(1-2) MNIST image recovery from measurement abs(Ax)+2tanh(abs(Ax))+eta
python mnist_main_mpr.py  --nonlinear-model 'abs(Ax)+2tanh(abs(Ax))+eta'  --num-outer-measurement-ls   200 300 400 500 600 700 800  --method-ls MPRS PPower MPRG_Step2 APPGD MPRG --noise-std-ls  0.0 0.01 0.1 0.5 1.0  

(1-3) MNIST image recovery from measurement 2sq(Ax)+3sin(abs(Ax))+eta
python mnist_main_mpr.py  --nonlinear-model '2sq(Ax)+3sin(abs(Ax))+eta'   --num-outer-measurement-ls   200 300 400 500 600 700 800   --method-ls MPRS PPower MPRG_Step2 APPGD MPRG --noise-std-ls  0.0 0.01 0.1 0.5 1.0 

(2) Run experiments on CelebA dataset 
(2-1)  CelebA image recovery from measurement abs(Ax+eta)+5tanh(abs(Ax))
python3 celebA_main_mpr.py   --nonlinear-model 'abs(Ax+eta)+5tanh(abs(Ax))' --num-outer-measurement-ls   2000 4000 6000 8000  --method-ls PPower APPGD MPRG --noise-std-ls 0.05 0.1 0.2 0.3 0.4 0.5 



## References

Large parts of the code are derived from [Bora et al.](https://github.com/AshishBora/csgm), [Hyder et al.](https://github.com/CSIPlab/appgd), [Liu et al.] (https://github.com/liuzq09/GenerativePCA)
