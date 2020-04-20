rm(list=ls()) # clear all workspace

### model corner
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(mlr3filters)
library(ranger)
library(mmpf)
library(mlr3viz)
library(precrec)
library(paradox)
library("mlr3tuning")
library(dataPreparation)

## get the task of different datasets with model tuning
# import data
completeData <- read.csv("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/oh_data/mice_oh_data.csv")

# transform the target y into factor type accordingly
completeData$y <- as.factor(completeData$y)

# can we use all features?
# Nope: delete those with too many levels as this would inflate the model?
# also kill the ID
train <- completeData
# [, -c( 
#   which(colnames(completeData) == "ID")
# )]

# to make the name for each column with unique name, so that there won't be any naming errors
colnames(train) <- make.names(colnames(train),unique = T)

# show only warning messages
lgr::get_logger("mlr3")$set_threshold("warn")

# choose a specific model(RF) and set the parameters
task <- TaskClassif$new(
  id="card_train", backend = train,
  target = "y"
)

### Tune the models
# we chose 3 main parameters to tune the model
# check available parameters
set.seed(2020)
learnerRF <-lrn("classif.ranger", predict_type = "prob")

# check for lower and upper bands for the three parameters
learnerRF$param_set

resamplingRF <- rsmp("cv", folds = 5)
measuresRF <- msrs(c("classif.auc"))

# 1--make parameter set
tuneRF_ps <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 1, upper = 500),
  ParamInt$new("mtry", lower = 2, upper = 10),
  ParamInt$new("min.node.size", lower = 10, upper = 50)
))

# 2--set terminator Eval/Combo/ClockTime/PerfReach/Stagnation
terminatorRF <- term("evals", n_evals = 100)

# 3--choose random search/ grid search
tunerSearchRF <- tnr("random_search")

# make autoTuner
at <- AutoTuner$new(
  learner = learnerRF,
  resampling = resamplingRF,
  measures = measuresRF,
  tune_ps = tuneRF_ps,
  terminator = terminatorRF,
  tuner = tunerSearchRF
)

#train the previous task with parameter setting
at$train(task)

# tuning result
at$tuning_result