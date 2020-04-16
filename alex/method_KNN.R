# KNN

" 
Performs k-nearest neighbor classification of a test set using a training set. 
For each row of the test set, the k nearest training set vectors are found. Make supervised classification for training functional data via k-nearest neighbors method. 
The method classifies the functional data to the group with the highest number of nearest neighbors. 
In case of tie the data is classified in the group with a shorter distance.
"

# glmnet::cv.glmnet()
"
<LearnerClassifKKNN:classif.kknn>
* Model: -
* Parameters: list()
* Packages: kknn
* Predict Type: response
* Feature types: logical, integer, numeric, factor, ordered
* Properties: multiclass, twoclass
"
# > learner$param_set$ids()
# [1] "k"        "distance" "kernel"   "scale" 
# https://rdrr.io/github/mlr-org/mlr3learners/src/R/LearnerClassifKKNN.R
# https://www.rdocumentation.org/packages/kknn/versions/1.3.1/topics/kknn

# k Number of neighbors considered.
# distance Parameter of Minkowski distance.
# kernel Kernel to use. Possible choices are "rectangular" (which is standard unweighted knn), "triangular", "epanechnikov" (or beta(2,2)), "biweight" (or beta(3,3)), "triweight" (or beta(4,4)), "cos", "inv", "gaussian", "rank" and "optimal".
# scale logical, scale variable to have equal sd.

library(mlr3)
library("mlr3viz")
library("precrec")
library("mlr3learners")

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
source("alex/read_data.R")
source("alex/train.R")

# tasks[["<type>"]][["<code>"]]


# define knn learn with cross validation (resampling)
learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=2)


resampling = rsmp("cv", folds = 5)

model <- train_model(tasks[["dl"]][["dummy"]], learner, resampling)
evaluate_result(model)

# train_all(tasks, learner, resampling)



