# Usage
Note: you may need a high-memory instance to run the code.

### Prepare data
- Create a soft link from /workspace/data to LOCAL_DATASET_DIR
```bash
ln -s LOCAL_DATASET_DIR /workspace/data
```

### Running instructions
For part A code, refer to the part A code. (some function API has minor changes.)
For part B code, simply run following instructions:
- For training, this would save checkpoints to folder 'train_check_point'
```bash
python main_train.py
```
- For inference and testing
```bash
python main_infer.py
```

### Data augmentation
In the training, we implemented data augmentation with horizontal flipping. i.e. Each image has 0.5 chances to flip and its annotations (masks + bboxes) will be changed accordingly.

### Expected Outputs
The expected outputs will be saving under directory 'testfig' and 'plotgt_results', in the current working directory.
 