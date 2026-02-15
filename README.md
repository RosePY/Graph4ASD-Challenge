# Graph4ASD-Challenge
The Graph4ASD Challenge aims to leverage the power of Graph Neural Networks (GNNs) for the classification of Autism Spectrum Disorder (ASD) using resting-state fMRI functional connectivity data from the ABIDE dataset.

## ABIDE Dataset
The Autism Brain Imaging Data Exchange (ABIDE) aggregates anonymized resting-state functional MRI (rs-fMRI) data acquired across 17 international sites. The dataset comprises brain network representations from 1,009 individuals, of whom 516 (51.14%) are diagnosed with autism spectrum disorder (ASD) and are treated as positive cases. Brain regions are defined according to the Craddock 200 atlas. For this challenge, a sample of the dataset was selected.

## Data Representation
This dataset represents each subject as a graph  
\[
G = (A, X)
\]

where:
- **A** → adjacency matrix (graph connectivity)
- **X** → node feature matrix


