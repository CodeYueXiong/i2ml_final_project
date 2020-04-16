# LDA
# Linear discriminant analysis (LDA) 線性判別分析

# MASS::lda()
"
<LearnerClassifLDA:classif.lda>
* Model: -
* Parameters: list()
* Packages: MASS
* Predict Type: response
* Feature types: logical, integer, numeric, factor, ordered
* Properties: multiclass, twoclass, weights
"
# learner$param_set$ids()
# [1] "prior"          "tol"            "method"         "nu"             "predict.method"
# https://rdrr.io/github/mlr-org/mlr3learners/src/R/LearnerClassifLDA.R

library(mlr3)
library("mlr3viz")
library("precrec")
library("mlr3learners")

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
source("alex/read_data.R")
source("alex/train.R")

# tasks[["<type>"]][["<code>"]]


# define lda learner with cross validation (resampling)
learner <- lrn("classif.lda", id = "lda")

resampling = rsmp("cv", folds = 3)

# dummy
#  Error in lda.default(x, grouping, ...) : 
# variable 9 appears to be constant within groups 

# oh
#  Error in lda.default(x, grouping, ...) : 
# variable 13 appears to be constant within groups 


# iv
#  Error in lda.default(x, grouping, ...) : 
# variable 8 appears to be constant within groups 

#  FLAG_MOBIL

model <- train_model(tasks[["dl"]][["oh"]], learner, resampling)
evaluate_result(model)

# train_all(tasks, learner, resampling)