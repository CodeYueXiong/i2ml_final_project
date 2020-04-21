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
completeData <- read.csv2("F:/1-Credit_Card_Approval_Prediction/i2ml_final_project/credit_card_prediction/iv_data/dl_iv_data.csv")

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

# choose a specific model(SVM) and set the parameters
task <- TaskClassif$new(
  id="card_train", backend = train,
  target = "y"
)

### Tune the models
# we chose 3 main parameters to tune the model
# check available parameters
set.seed(2020)
learnerSVM <-lrn("classif.svm", predict_type = "prob")

# check for lower and upper bands for the three parameters
learnerSVM$param_set

resamplingSVM <- rsmp("cv", folds = 5)
measuresSVM <- msrs(c("classif.auc"))

# 1--make parameter set
tuneSVM_ps <- ParamSet$new(list(
  ParamFct$new("kernel", levels = "polynomial"),
  ParamInt$new("gamma", lower = 5, upper = 20)
))

## levels = c("linear", "polynomial", "radical", "sigmoid")),

# 2--set terminator Eval/Combo/ClockTime/PeSVMReach/Stagnation
terminatorSVM <- term("evals", n_evals = 5)

# 3--choose random search/ grid search
tunerSearchSVM <- tnr("random_search")

# make autoTuner
at <- AutoTuner$new(
  learner = learnerSVM,
  resampling = resamplingSVM,
  measures = measuresSVM,
  tune_ps = tuneSVM_ps,
  terminator = terminatorSVM,
  tuner = tunerSearchSVM
)
# record time
classiftime1 <- proc.time()

#train the previous task with parameter setting
at$train(task)

# tuning result
at$tuning_result

# time consumed in tuning
classiftime <- proc.time()-classiftime1

# print duration of time
print(classiftime)