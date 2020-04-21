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

## import iv data group
dl_iv_data <- read.csv2("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/iv_data/dl_iv_data.csv")
mf_iv_data <- read.csv2("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/iv_data/mf_iv_data.csv")
mice_iv_data <- read.csv2("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/iv_data/mice_iv_data.csv")

## import one-hot data group
dl_oh_data <- read.csv("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/oh_data/dl_oh_data.csv")
mf_oh_data <- read.csv("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/oh_data/mf_oh_data.csv")
mice_oh_data <- read.csv("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/oh_data/mice_oh_data.csv")

# transform the target y into factor type accordingly for all
dl_iv_data$y <- as.factor(dl_iv_data$y)
mf_iv_data$y <- as.factor(mf_iv_data$y)
mice_iv_data$y <- as.factor(mice_iv_data$y)
dl_oh_data$y <- as.factor(dl_oh_data$y)
mf_oh_data$y <- as.factor(mf_oh_data$y)
mice_oh_data$y <- as.factor(mice_oh_data$y)


#create tasks for all the data group
task = list(TaskClassif$new("dl_iv_lg", backend= dl_iv_data , target = "y"),
            TaskClassif$new("mf_iv_lg", backend= mf_iv_data , target = "y"),
            TaskClassif$new("mice_iv_lg", backend= mice_iv_data , target = "y"),
            TaskClassif$new("dl_oh_lg", backend= dl_oh_data , target = "y"),
            TaskClassif$new("mf_oh_lg", backend= mf_oh_data , target = "y"),
            TaskClassif$new("mice_oh_lg", backend= mice_oh_data , target = "y")
)

# create a benchmark
design = benchmark_grid(
  tasks = task,
  learners = lrn("classif.ranger", predict_type = "prob"),
  resampling = rsmp("cv", folds=5L)
)

# run the benchmark
bmr_lg <- benchmark(design)
# save the results
result <- bmr_lg$aggregate(measures = msr("classif.auc"))
# print result
print(result)
