D-BCD
==
  
Code for [Personalized On-Device E-health Analytics with Decentralized Block Coordinate Descent ](https://arxiv.org/abs/2112.09341).

Abstract
-
Actuated by the growing attention to personal healthcare and the pandemic, the popularity of E-health is proliferating.  
Nowadays, enhancement on medical diagnosis via machine learning models has been highly effective in many aspects of e-health analytics. Nevertheless, in the classic cloud-based/centralized e-health paradigms, all the data will be centrally stored on the server to facilitate model training, which inevitably incurs privacy concerns and high time delay. Distributed solutions like Decentralized Stochastic Gradient Descent (D-SGD) are proposed to provide safe and timely diagnostic results based on personal devices. However, methods like D-SGD are subject to the gradient vanishing issue and usually proceed slowly at the early training stage, thereby impeding the effectiveness and efficiency of training. In addition, existing methods are prone to learning models that are biased towards users with dense data, compromising the fairness when providing E-health analytics for minority groups. 

In this paper, we propose a Decentralized Block Coordinate Descent (D-BCD) learning framework that can better optimize deep neural network-based models distributed on decentralized devices for E-health analytics. As a gradient-free optimization method, Block Coordinate Descent (BCD) mitigates the gradient vanishing issue and converges faster at the early stage compared with the conventional gradient-based optimization. To overcome the potential data scarcity issues for users' local data, we propose similarity-based model aggregation that allows each on-device model to leverage knowledge from similar neighbor models, so as to achieve both personalization and high accuracy for the learned models. Benchmarking experiments on three real-world datasets illustrate the effectiveness and practicality of our proposed D-BCD, where additional simulation study showcases the strong applicability of D-BCD in real-life E-health scenarios.


Cite Our Paper
-
```
@article{ye2022personalized,
  title={Personalized on-device e-health analytics with decentralized block coordinate descent},
  author={Ye, Guanhua and Yin, Hongzhi and Chen, Tong and Xu, Miao and Nguyen, Quoc Viet Hung and Song, Jiangning},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  publisher={IEEE}
}
```
