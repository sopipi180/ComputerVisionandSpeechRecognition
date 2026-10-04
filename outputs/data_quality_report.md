# Data-quality report (Milestone 1)

- images: 10,480   lesions: 5,240

## Missing values and what we do with them

| column | % missing | decision |
|---|---|---|
| anatom_site_special | 98.0 | drop - 98 % missing |
| diagnosis_4 | 85.5 | not used as input (leaky) |
| melanocytic | 77.2 | not used as input (leaky) |
| anatom_site_general | 37.3 | fill with 'unknown' - being unrecorded is itself informative |
| diagnosis_3 | 1.5 | not used as input (leaky) |
| age_approx | 0.4 | impute with the TRAIN median (a real value we would also have at prediction time) |

## Label consistency

- lesions with exactly 2 images: 5,240 of 5,240
- lesions with one image of each type: 5,240 of 5,240
- lesions in metadata but not in training_gt: 0
- lesions with more than one positive class: 0
- 11-classes that map to several diagnosis_1 values: ['AKIEC']

## Shortcut check (acquisition fields vs the label)

### image_manipulation (row %)

| image_manipulation   |   Benign |   Indeterminate |   Malignant |
|:---------------------|---------:|----------------:|------------:|
| altered              |     39.7 |            12.8 |        47.5 |
| instrument only      |     27.9 |             2   |        70.1 |

### image_type (row %)

| image_type         |   Benign |   Indeterminate |   Malignant |
|:-------------------|---------:|----------------:|------------:|
| clinical: close-up |     28.3 |             2.3 |        69.4 |
| dermoscopic        |     28.3 |             2.3 |        69.4 |

Conclusion: image_type is identical across classes by construction (every lesion has one of each), so it carries no label information, but it does change image statistics. image_manipulation does differ between classes, so it is treated as an acquisition shortcut and is not used as a model input.

## Columns NOT used as model inputs (label leakage)

- `diagnosis_2` - finer level of the diagnosis
- `diagnosis_3` - finer level of the diagnosis
- `diagnosis_4` - finer level of the diagnosis
- `dx11` - the 11-class label itself
- `melanocytic` - derived from the diagnosis
- `diagnosis_confirm_type` - how the diagnosis was confirmed (biopsy = already suspicious)
- `concomitant_biopsy` - whether a biopsy was taken (already suspicious)
- `image_manipulation`, `attribution`, `copyright_license` - acquisition / provenance, not the lesion