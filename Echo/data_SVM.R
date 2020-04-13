rm(list=ls()) # clear all workspace
# import data
source("/Users/echo/Desktop/Echo/data_prep_one_hot_encoding.R")

# install.packages("mlr3")
# install.packages("mlr3learners")
# install.packages("sjlabelled")
# install.packages("sjmisc")
# install.packages("magrittr")
# install.packages("dplyr")
# install.packages("frequency")
# install.packages("R6")
install.packages("e1071")

library(mlr3)
library(mlr3learners)
# library(R6)

library(sjlabelled)
library(sjmisc)
library(magrittr)

library(dplyr)

library(frequency)
library("e1071")

# omit NA data
data_onehot <- na.omit(data_onehot)

data_onehot$y <- as.factor(data_onehot$y)



colnames(data_onehot) <- make.names(colnames(data_onehot),unique = T)

glimpse(data_onehot)

# directly using SVM
attach(data_onehot)

x <- subset(data_onehot, select=-y)
y <- y

svm_model <- svm(y ~ ., data=data_onehot, scale = FALSE)
summary(svm_model)

# Run Prediction and you can measuring the execution time in R
pred <- predict(svm_model,x)
system.time(pred <- predict(svm_model,x))

# See the confusion matrix result of prediction, using command table to compare the result of SVM prediction and the class data in y variable.
table(pred,y)

# TUning SVM to find the best cost and gamma ..
svm_tune <- tune(svm, train.x=x, train.y=y, 
                 kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))

print(svm_tune)

# After you find the best cost and gamma, you can create svm model again and try to run again
svm_model_after_tune <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
summary(svm_model_after_tune)

# Run Prediction again with new model
pred <- predict(svm_model_after_tune,x)
system.time(predict(svm_model_after_tune,x))
# see the confusion matrix
table(pred,y)

# create learners and tasks for mlr3
task_approval <- TaskClassif$new(id = "data_onehot", backend = data_onehot, target = "y") # how to solve the problem re the number of 1 is too small
# load SVM learner
learner = mlr3::lrn("classif.svm")
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
