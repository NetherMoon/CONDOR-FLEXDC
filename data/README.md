# Data Collection Overview

Our data was collected with the simulater used for the paper "HPC data center participation in demand response: an adaptive policy with QoS assurance"  Zhang, Wilson, et al. 21. This work introduced the AQA (Adaptive Quality Assurance) demand response framework, which this work builds off of. 

That work used a bespoke data center computational simulator, wihch we colloquially call the AQA simulator. Currently, this simulator is not open-sourced, but there is work at Boston University's PEACLAB to improve the codebase and open-source the simulator. Check the PEACLAB website to see if that is completed.

## Collecting Data from the AQA Simulator 

The AQA simulator, when using the gradient descent procedure described in the Zhang et al. 21 paper above, produces files that record each gradient descent step, including the P/R/W values and other information. These are of the format "GD{experiment name}.csv". 

We take these files, turn them into a single Pandas dataframe, and save them as a CSV. This is then used by our PyTorch DataSet/DataLoader during training. 

The data_wrangling.ipynb file shows this full process. If you want to train on your own data or add new data, follow the example in that file. This will produce a file called "all_data.csv" which you can then use with our dataloader. 

## Workload Mix Handling

One of the features that is not explicitly described in the GD records described above is the workload mix (set of job types) being used for the given GD run. To alleviate this, one of the things that is required by the method that collects and transforms the data into the final format being used is the name of the workload mix, and the location of a dictionary containing information about the workload mixes. 

I will not describe exactly how these are saved in detail, but the top-level takeaway is that this dictionary is made by the workload_mix_db_handling.ipynb notebook. See that for more details, the notebook also includes instructions on how to modify it to include your own mixes, job types, or both. Then, you can use your new/modified mixes in the data collection script. 
