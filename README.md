# PyTorch-TTUHPCC
How to set up and run code on the HPCC at TTU. In my example, I use a high energy physics dataset and a convolutional autoencoder to distinguish HtoBB from ZtoQQ. 

Step 1. Creating our conda environment:

- Run this command 


  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh | bash
  
- This should create a file that looks like this in your directory

  
![image](https://github.com/user-attachments/assets/4822257a-6253-46da-a45e-43b02daed4ac)

- Run this file using the command:

  bash Miniforge3-Linux-x86_64.sh

- From this you will need to route to your directory where conda.sh is stored. In my case I set my current directory to

  HOME/ENTER/etc/profile.d/conda.sh

  manually using this command:

  ![image](https://github.com/user-attachments/assets/986ab85f-d873-46f7-8e3d-e5f2ef346ae9)

- Now you have to actually create the conda environment we are going to use for machine learning. While typically it would be appropriate to install a relatively new version of python, we cannot do that in this case. Instead, we will install python 3.8.20, which works with the current version of cuda across the entire system cluster(11.0)

![image](https://github.com/user-attachments/assets/2e71e697-7662-4f6f-9f33-1e90fa5ed4cd)

Step 2: Activating our environment, and installing needed packages. 






  
 
  

  

  

  


