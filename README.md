# InceptionDTA
InceptionDTA: Predicting Drug-Target Binding Affinity with Biological Context Features and Inception Networks
## Data
We utilized three DTA datasets including Davis, Kiba and PDBbind refine set. Davis and Kiba datasets were downloaded from [here](https://github.com/hkmztrk/DeepDTA/tree/master/data). PDBbind dataset was downloaded from [here](http://www.pdbbind.org.cn/download.php). It should be noted that you should register and login before downloading data files from the PDBbind repositories.
<br/>
Each dataset folder includes binding affinity (i.e. Y), protein sequences (i.e. proteins.txt), drug SMILES (i.e. ligands_can.txt), and a folder includes the train and test folds settings (i.e. folds).

## Requirements
Python <br/>
Tensorflow <br/>
Keras <br/>
Numpy <br/>

## Usage
For training and evaluation of the method, you can run the following script.
```
python run_experiments.py --num_windows 128 64 32 \
                          --batch_size 256 \
                          --num_epoch 300 \
                          --max_seq_len 1000 \
                          --max_smi_len 85 \
                          --dataset_path 'data/davis/' \
                          --problem_type 1 \
                          --is_log 1 \
                          --log_dir 'logs/' \
```

## Cold-start
Under the constraints of cold-start, InceptionDTA can only predict binding affinity from unseen protein, unseen drug and both of them. <br/>
To train protein cold-start change value of problem_type to 2, to train drug cold-start change value of problem_type to 3 and to train protein-drug cold-start change value of problem_type to 4. For example you can use the following script to train protein cold-start:
```
python run_experiments.py --num_windows 128 64 32 \
                          --batch_size 256 \
                          --num_epoch 300 \
                          --max_seq_len 1000 \
                          --max_smi_len 85 \
                          --dataset_path 'data/davis/' \
                          --problem_type 2 \
                          --is_log 1 \
                          --log_dir 'logs/' \
```
Also, an alternative splitting setting based on the physio-chemical properties of compound molecules for the PDBbind dataset is considered. These properties include logP values computed with Open Babel logP and XLOGP3 tools.
To test based on the Open Babel logP, change the value of 'problem_type' to 2. To test based on XLOGP3, change it to 3.
```
python run_experiments.py --num_windows 128 64 32 \
                          --batch_size 256 \
                          --num_epoch 300 \
                          --max_seq_len 1000 \
                          --max_smi_len 200 \
                          --dataset_path 'data/pdb/' \
                          --problem_type 2 \
                          --is_log 0 \
                          --log_dir 'logs/' \
```
