# IIB_project_code
## Overview
This is the code repository for my masters project titled **Improvements on the Linearized Laplace Method**
## Sampling
There is code for two methods to sample from a Gaussian posterior distribution, the first, proposed [here](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Farxiv.org%2Fabs%2F2210.04994&data=05%7C01%7Ctr452%40universityofcambridgecloud.onmicrosoft.com%7Ccff4a85031ae4beb43e008daac6c12aa%7C49a50445bdfa4b79ade3547b4f3986e9%7C0%7C0%7C638011877405786205%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=z8eZff%2BH58cpxnFNmaqIx6RTaFjGfvBeI7ZHp3DDC74%3D&reserved=0) 
uses an EM algorithm to optimise the prior such that the posterior best predicts the data, this is achieved by sampling from the proposed posterior at each E step by 
optimising a least squares problem with injected noise, the samples can then be used to compute a new prior covariance that converges towards the optimum. The second
method is known as stochastic gradient hamilton monte carlo, proposed [here](https://eur03.safelinks.protection.outlook.com/?url=https%3A%2F%2Farxiv.org%2Fabs%2F1402.4102&data=05%7C01%7Ctr452%40universityofcambridgecloud.onmicrosoft.com%7Ca8658a13077f42d106cf08dab836b5c1%7C49a50445bdfa4b79ade3547b4f3986e9%7C0%7C0%7C638024842788212083%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C3000%7C%7C%7C&sdata=j2N3tQMIbLh%2F4XxW%2FXJfdKD5dKeXPYBBFw7NH8SnSnM%3D&reserved=0)
which expands on traditional HMC by considering a friction term to account for the effects of stochastic gradient noise. 
