# Data used for Wing-Segmentation

The images used for training this model were sourced from the following collections:
1. [Heliconius Collection (Cambridge Butterfly)](#heliconius-collection-(cambridge-butterfly))
2. [Monteiro Dataset](#monteiro-data)
3. [Smithsonian Tropical Research Institute (STRI) Data](#smithsonian-tropical-research-institute-(stri)-data)

## [Heliconius Collection (Cambridge Butterfly)](https://huggingface.co/datasets/imageomics/Heliconius-Collection_Cambridge-Butterfly)

394 dorsal images of _Heliconius erato_ and _Heliconius melpomene_ butterflies were sourced from this subset of Chris Jiggins' research group's collection from the University of Cambridge.

![species distribution chart for CB data colored by label](https://github.com/user-attachments/assets/0634d0c9-ab08-4f0c-868a-7c34a52ab513)


### How to Access*

Install the [cautious-robot](https://github.com/Imageomics/cautious-robot) package.

Run
```bash
cautious-robot -i <path/to>/wing_seg_heliconius_collection.csv -o <path/to/output-directory> -s label -u source_url -l 256 -v "md5"
```

This will download all images into subfolders based on their label (`damaged`, `dorsal`, `body_attached`, or `incomplete`) and do the same for 256 x 256 downsized copies of the images (in a separate parent folder). It will then verify a match of all the full-sized image MD5s using the [sum-buddy package](https://github.com/Imageomics/sum-buddy).

### Citation
Full bibtex citations are provided in [Heliconius_collection_cambridge.bib](/heliconius_collection_cambridge.bib): these are for both the compilation and all original image sources from the Butterfly Genetics Group at University of Cambridge.

## [Monteiro Data](https://lepdata.org/)

Collection of 199 full body images of 105 species of butterflies across 16 genera from the Monteiro Lab (a subset of their collection).

![Genus distribution chart for Monteiro data colored by label](https://github.com/user-attachments/assets/8c1c4e16-aa36-44db-9bd3-48e33d791c64)


### How to Access*

Install the [cautious-robot](https://github.com/Imageomics/cautious-robot) package.

Run
```bash
cautious-robot -i <path/to>/wing_seg_monteiro.csv -o <path/to/output-directory> -s label -u source_url -l 256 
```

This will download all images into subfolders based on their label (in this case, just `body_attached`) and do the same for 256 x 256 downsized copies of the images (in a separate parent folder). 


### Citation

Images were accessed from the [Monteiro Lab website](https://lepdata.org/monteiro/).

```
@article{article,
  author = {Silveira, Margarida and Monteiro, Antonia},
  year = {2008},
  month = {11},
  pages = {130-6},
  title = {Automatic recognition and measurement of butterfly eyespot patterns},
  volume = {95},
  journal = {Bio Systems},
  doi = {10.1016/j.biosystems.2008.09.004}
}
```

## [Smithsonian Tropical Research Institute (STRI) Data](https://huggingface.co/datasets/imageomics/STRI-Samples)

This is a subset of 207 images of butterfly wings collected by [Owen McMillan](https://stri.si.edu/scientist/owen-mcmillan) and members of his lab at the [Smithsonian Tropical Research Institute (STRI)](https://stri.si.edu/). There are 11 speices represented, primarily from the genus Heliconius, but also including the following genera: Junonia, Eueides, Neruda, and Dryas.


![species distribution chart for STRI data colored by label](https://github.com/user-attachments/assets/92da2996-b4f0-462a-b666-4ea715669f3b)


### How to Access*

These images are available for direct download from the [STRI-Samples Hugging Face Dataset](https://huggingface.co/datasets/imageomics/STRI-Samples).

As with the other two, images should be downloaded into folders based on their labels: `damaged`, `dorsal`, or `incomplete`, and then downsized with [this script]() to 256 x 256.

### Citation

Christopher Lawrence, Owen McMillan, Daniel Romero, Carlos Arias. (2023). Smithsonian Tropical Research Institute (STRI) Samples. Hugging Face. https://huggingface.co/datasets/imageomics/STRI-Samples.


*Note that all images could be downloaded using the `cautious-robot` package from the combined CSV [`wing_segmentation_images.csv`](/data/wing_segmentation_images.csv); the automated verification of checksums just would not be available.
