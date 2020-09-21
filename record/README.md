# Week 1

1. Data download and cleaning `src/preprocessing.py`

	a. Parsing datetime column

	b. Splitting into **training** (*pre 2010*) and **testing** (*post 2010*)

	c. Storing into the .h5 format

2. Building pipeline `src/pipeline.py`

	a. Building the **SplitDate()** pipeline architecture 

	b. Building the SlidingWindow pipeline for both X and y

	c. Setting up the total pipeline for all the data

3. Training and Testing Models `src/pipeline.py`
	
	a. Linear Regression

	b. Decision Trees

	c. Multi-Layer Perceptron

	d. Random Forest (haven't completed running this one)
