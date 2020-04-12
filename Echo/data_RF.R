rm(list=ls()) # clear all workspace
# import data
source("/Users/echo/Desktop/Echo/data_prep_one_hot_encoding.R")

install.packages("ranger")

library(mlr3)
library(mlr3learners)
library(ranger)

# omit NA data
data_onehot <- na.omit(data_onehot)

data_onehot$y <- as.factor(data_onehot$y)

colnames(data_onehot) <- make.names(colnames(data_onehot),unique = T)

glimpse(data_onehot)

# create learners and tasks for mlr3
task_approval <- TaskClassif$new(id = "data_onehot", backend = data_onehot, target = "y") # how to solve the problem re the number of 1 is too small
# load Random Forest learner
learner = mlr_learners$get("classif.ranger")
# print(learner)

## Basic train + predict

# train/test split
train_set <- sample(task_approval$nrow, 0.8 * task_approval$nrow)
test_set <- setdiff(seq_len(task_approval$nrow), train_set)

# freq(data_onehot$FLAG_MOBIL)
# train the model
learner$train(task_approval, row_ids = train_set)
# probelm: when there is only constant in a variable, like FLAG_MOBIL, how to set it so as for scale = FALSE
# predict data
prediction <- learner$predict(task_approval, row_ids = test_set)

# calculate performance
prediction$confusion
