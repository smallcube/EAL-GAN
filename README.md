
EAL-GAN: Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning
==
This is the official implementation of “Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning”. Our paper has been submitted to the IEEE for possible publication, and a Preprint version of the manuscript can be found in Arxiv. If you use the codes in this repo, please cite the paper as follows:<br>

> @misc{chen2021supervised,<br>
          &nbsp;&nbsp;&nbsp;&nbsp; title={Supervised Anomaly Detection via Conditional Generative Adversarial Network and Ensemble Active Learning}, <br>
        &nbsp;&nbsp;&nbsp;&nbsp;author={Zhi Chen and Jiang Duan and Li Kang and Guoping Qiu},<br>
      &nbsp;&nbsp;&nbsp;&nbsp;year={2021},<br>
      &nbsp;&nbsp;&nbsp;&nbsp;eprint={2104.11952},<br>
      &nbsp;&nbsp;&nbsp;&nbsp;archivePrefix={arXiv},<br>
      &nbsp;&nbsp;&nbsp;&nbsp;primaryClass={cs.LG}<br>
}

&nbsp;&nbsp;&nbsp;&nbsp;We have implemented two versions of the proposed EAL-GAN, including:(1) one for classical anomaly detection datastes, whose codes are in the "EAL-GAN" folder, and (2) one for the image datasets, whose codes are in the "EAL-GAN-image" folder. Please note the EAL-GAN-image is sensitive to the learning rate and initiation. We print the changes in the loss of discriminators and generator during training, and if the loss become NAN in the first epoch, you should re-run the code with smaller learning rate.  If that doesn't happen, you can expect a promising result.


Requirements
===
Pytorch >1.6 <br>
Python 3.7<br>

Getting started
===
(1)	You can run the script “train_EAL_GAN.py” to train the model proposed in our paper.<br>
(2)	Some of the datasets in our paper are given in the folder “/data”. <br>
(3)	Models/EAL-GAN.py is the proposed model.<br>
(4)	Models/losses.py is the loss functions.<br>


Acknowledgments
==
some of our codes (e.g., Spectral Normalization) are extracted from the [PyTorch implementation of BigGAN]( https://github.com/ajbrock/BigGAN-PyTorch).
