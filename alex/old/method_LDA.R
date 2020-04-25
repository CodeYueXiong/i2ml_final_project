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
library(mlr3viz)
library(precrec)
library(mlr3learners)

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
source("alex/read_data.R")
source("alex/train.R")

# tasks[["<type>"]][["<code>"]]


# define lda learner with cross validation (resampling)
learner <- lrn("classif.lda", id = "lda", predict.type="prob")

resampling = rsmp("cv", folds = 5)

model <- train_model(tasks[["dl"]][["dummy"]], learner, resampling)
model$score(msr("classif.acc"))
model$prediction()$confusion
#evaluate_result(model)

# train_all(tasks, learner, resampling)