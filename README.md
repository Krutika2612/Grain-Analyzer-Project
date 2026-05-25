# Grain-Analyzer-Project
An automated Computer Vision and Machine Learning pipeline for metallurgical grain boundary detection, segmentation, and microstructural grain size analysis.

Project Overview This repository contains the codebase for the Automated Microstructural Grain Analyzer, developed as part of an IIT Bombay internship project. Traditional grain analysis requires manual counting and boundary tracing, which is time-consuming and subjective. This project automates the process by deploying an advanced computer vision model to instantly detect grain boundaries, segment individual grains, and compute essential metallurgical metrics from digital micrographs.

This tool is designed to assist materials scientists and quality control engineers in rapidly assessing the microstructural integrity and mechanical properties of metal samples.

Key Features

Automated Boundary Detection: Utilizes advanced computer vision techniques to accurately map distinct grain boundaries, even in images with uneven lighting or noise.

Instance Segmentation: Isolates individual grains to calculate microstructural properties, such as average grain area and size distribution.

Robust Preprocessing Pipeline: Implements sophisticated image filtering (e.g., Gaussian blurring, adaptive thresholding) to enhance structural contrast before feeding the images into the detection algorithm.

Quantitative Output: Transforms visual data into actionable numerical metrics, drastically reducing the time required for metallurgical characterization.

Dataset Information

Source Images: The dataset comprises high-resolution optical and/or scanning electron microscope (SEM) images of metal alloys.

Image Characteristics: These micrographs capture distinct crystalline structures, grain boundaries, and varying phases inherent to the material.

Preprocessing: Images undergo standard scaling, normalization, and contrast enhancement (such as CLAHE - Contrast Limited Adaptive Histogram Equalization) to ensure the features are optimally visible for the model.

Technology Stack

Computer Vision: Python, OpenCV (cv2), Scikit-Image.

Data Processing & Analytics: NumPy, Pandas, Matplotlib (for plotting grain size distributions).

Deep Learning/Modeling (Optional - adjust as needed): TensorFlow/Keras (if a segmentation model like U-Net was used) or traditional ML segmentation algorithms.

Environment: Google Colab / Jupyter Notebooks.

Credits Prepared by IIT Bombay Interns -  Krutika Kamble and Devraj Mane .
