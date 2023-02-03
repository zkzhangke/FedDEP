# FedDEP
Here are the source files for FedDEP. Our implementation is built on FederatedScope [1].

To run our FedDEP pipeline demo on Cora dataset with 5 clients, please follow these four steps.

1. Please download this project to your directory `path/to/FedDEP/`;

2. Please open `path/to/FedDEP/FederatedScope/federatedscope/main.py` and modify the `path/to/FedDEP/FederatedScope/` in line 3 to your actual path;
 
3. Please open `path/to/FedDEP/FederatedScope/federatedscope/main.py` and modify the `path/to/FedDEP/FederatedScope/` in line 3 to your actual path;

4. Type the command `cd path/to/FedDEP/FederatedScope/federatedscope`;

5. You may need to create the environment to run FedDEP via the command `conda env create --file path/to/FedDEP/FederatedScope/environment_packages.txt`

6. After finish the environment configuration, you can run FedDEP with command `python main.py --cfg feddep_on_cora5.yaml`.

You may modify different values in the ".yaml" file to simulate different scenarios. 

For more instructions of FederatedScope, please refer to [https://federatedscope.io](https://federatedscope.io)

[1] Wang, Zhen, et al. "Federatedscope-gnn: Towards a unified, comprehensive and efficient package for federated graph learning." Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2022.
