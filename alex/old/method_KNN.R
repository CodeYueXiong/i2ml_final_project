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

# k         Number of neighbors considered.
# distance  Parameter of Minkowski distance.
# kernel    Kernel to use. Possible choices are "rectangular" (which is standard unweighted knn), "triangular", "epanechnikov" (or beta(2,2)), "biweight" (or beta(3,3)), "triweight" (or beta(4,4)), "cos", "inv", "gaussian", "rank" and "optimal".
# "rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal"
# scale     logical, scale variable to have equal sd.

# k         big impact in AUC
# distance  small improvement in AUC (significant increase in runtime)

library(mlr3)
library(mlr3viz)
library(precrec)
library(mlr3learners)
library(mlr3tuning)

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
source("alex/read_data.R")
source("alex/train.R")

## ----------- nested TEST

task = tasks$dl$dummy
resampling = rsmp("holdout")
learner = lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=1, scale=FALSE)
measures = msr("classif.auc")

param_set = paradox::ParamSet$new(
  params = list(paradox::ParamInt$new("k", lower = 10, upper = 20)))

terminator = term("evals", n_evals = 5)
tuner = tnr("grid_search", resolution = 10)

at = AutoTuner$new(learner, resampling, measures = measures,
                   param_set, terminator, tuner = tuner)

resampling_outer = rsmp("cv", folds = 3)
rr = resample(task = task, learner = at, resampling = resampling_outer)

## -----------------------


# tasks[["<type>"]][["<code>"]]
kernal_type <- c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal")

# define knn learn with cross validation (resampling)
resampling = rsmp("cv", folds = 5)

# feature selection is important

# for(k in kernal_type){
#   task <- tasks$dl$dummy$select(c("amt_income_total", "days_employed", "flag_own_car_y", "flag_own_realty_y"))
#   learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=2, kernel=k, scale=TRUE)
#   model <- train_model(task, learner, resampling)
#   cat(k, model$aggregate(msr("classif.auc")), "\n")
# }

#learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=2, kernel=kernal_type[1], scale=FALSE)
learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=1, scale=FALSE)

model <- train_model(tasks[["dl"]][["dummy"]], learner, resampling)
model$score(msr("classif.acc"))
model$prediction()$confusion
#evaluate_result(model)

models <- train_all(tasks, learner, resampling)
evaluate_models(models)




