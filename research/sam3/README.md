# SAM3 Research 

Testing SAM3 though the HuggingFace transformers library for the butterfly wing segmentation. 

## Functionality
- Runs SAM3 on the butterfly images that are calibrated by using the prompt "butterfly wing" 
- Detects all the 4 wings (forewing left(FW_left), forewing right(FW_right), hindwing left(HW_left), hindwing right(HW_right)) and labels and drawns an outline for them after detection. 
- The labelling is based on mask area and position.

## Usage
```python
python test_sam3_batch.py
```

## Notes 
- Tested on 21 images from the STRI Amanda dataset
- Issue #10
