# Online Learning for Vehicle Trajecory Planning

## Introduction 
This program provides the dataset for temporal sequence of vehicle trajectory for online learning algorithms.  
The example dataset is form Argoverse forcasting and the user can make customized dataset using this framework.

## Installation
To use Argoverse dataset, install Argoverse Map and download the dataset files from github and website.  
Github code: https://github.com/argoverse/argoverse-api  
Website: https://www.argoverse.org/av1.html  

Install packages by 
``` pip install -e . ```

## arguments and commands
1. Save training set:  
```
python main.py --do_train --data_dir './Argoverse11/forecasting/forecasting_train_v1.1/train/data/' --'output_dir' './outputs/ --temp_file_name 'data_temp' --temp_file_dir './temp_files' --core_num 8
```

2. Load(reuse) training set:
Add ```--reuse_temp_file ``` to the command line. Then it will read the saved dataset.


3. Iterate the evaluation files and create sequential cases  
We recommand the user to iterate cases one by one. Example code is:

```
python main.py --do_train --data_dir_for_val './Argoverse11/forecasting/forecasting_train_v1.1/train/data/' --'output_dir' './outputs/ --temp_file_name 'data_temp' --temp_file_dir './temp_files' --core_num 8
```


## Other parameters
DenseTNT used special features in the Argoverse dataset. Details are introduced in https://github.com/Tsinghua-MARS-Lab/DenseTNT

We can enable them int other paramters by:  
``` --other_parameters xxx, xxx ``` where xxx could be: 
```
direction, set_predict, goals_2D, stage_one, semantic_lane, enhance_rep_4, subdivide
```