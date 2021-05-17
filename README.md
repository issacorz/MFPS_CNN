# MFPS_CNN
This is the public site for the paper under submission named: "Paper Name"


# LIBRARY REQUIREMENTS

	We will need to install some basic packages to run the programs as followed:
	
		+ git version 2.30.0.windows.2
		
		+ python 3.7.6
		
		+ numpy 1.16.0
		
		+ pandas 1.0.3
		
		+ imblearn 0.7.0

            + tensorflow-gpu 1.14.0
# USAGE
Clone the repository or download compressed source code files.

Extract the data.rar in data and run ion_data_preprocessing.py which will generate train and test file.


	$ git clone https://github.com/issacorz/MFPS_CNN.git
	$ cd MFPS_CNN

You can see help option by using:

	$ python run.py --help
	
Example for setting parameter


	$ python run.py \
	--num_windows [256, 256, 256] \ 
	--window_lengths [2, 4, 16] \
	--num_hidden 1000 \
	--batch_size 100 \
	--keep_prob 0.7 \ 
	--learning_rate 0.001 \
	--regularizer 0.001 \ 
	--max_epoch 100 \
	--seq_len 4000 \ 
	--num_classes 2 \ 
	--log_interval 100 \ 
	--save_interval 100 \ 
	--log_dir './logs' \  
	--test_file 'your test file direction' \ 
	--train_file 'your train file direction' 
