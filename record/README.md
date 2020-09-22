# Week 2

#### Day 2 - (*To Do*)

1. Functionify and classify code to make everything easier

2. Do the rest of the ML algorithms that can be done, see how low of an error you can get

3. Start up on ARIMA (a new file completely)

#### Day 1
1. Implemented **mini-batch learning architecture** . Much harder then it seems, sklearn doesn't really help at all here

	a. Loading the data via batches

	b. Fitting the pipeline in a roundabout manner

	c. Mini-batch learning **linear classifier** (used SGDClassifier, seems like OLS per batch isn't doable?, ask about this)

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
