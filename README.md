EAL-GAN: Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning
===
This is the official implementation of “Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning”. Our paper has been submitted to the IEEE for possible publication, and a Preprint version of the manuscript can be found in Arxiv. If you use the codes in this repo, please cite the paper as follows:<br>
[1]	Zhi Chen, Jiang Duan, Li Kang, and G. Qiu, "Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning," arXiv, , 2021.


Requirements
===
Pytorch >1.6 <br>
Python 3.7<br>

Getting started
===
(1)	You can run the script “train_CBG_AN.py” to train the model proposed in our paper.<br>
(2)	Some of the datasets in our paper are given in the folder “/data_mat”. <br>
(3)	Models/CB-GAN.py is the proposed model.<br>
(4)	Models/losses.py is the loss functions.<br>


Acknowledgments
===
some of our codes (e.g., Spectral Normalization) are extracted from the [PyTorch implementation of BigGAN]( https://github.com/ajbrock/BigGAN-PyTorch).
