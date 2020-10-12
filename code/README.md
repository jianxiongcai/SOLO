# Usage
Note: you may need a high-memory instance to run the code.

### Prepare data
- Create a soft link from /workspace/data to LOCAL_DATASET_DIR
```bash
ln -s LOCAL_DATASET_DIR /workspace/data
```

### Running instructions
For part A code, simply run following instructions:
```bash
cd CODE_DIRECTORY
python dataset.py
python solo_head.py
```

### Expected Outputs
The expected outputs will be saving under directory 'testfig' and 'plotgt_results', in the current working directory.
 