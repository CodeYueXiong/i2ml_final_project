# import data
setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
source("credit_card_prediction/data_preparation.R")
data <- final_data

# check type of all colums
str(data)

# ------------------------ data prep: dummy variable
library(fastDummies)
data$y <- as.numeric(as.character(data$y)) # avoid "y" (factor column) be transformed to dummy variable.

# transform all factor variables to dummy variables, and removes the columns used to generate the dummy columns.
data_dummy <- fastDummies::dummy_cols(data, remove_selected_columns=TRUE) 
data_dummy$y <- as.factor(data_dummy$y)

 
# transfor all dummy variables to factors
# data_dummy <- data_dummy %>% mutate_if(is.integer, as.factor)

# column name convention fix
library(janitor)
data_dummy <- clean_names(data_dummy)

# remove NA, since mlr3 cannot work with NA
library(dplyr)
data_dummy <- data_dummy %>% mutate_if(is.integer, ~replace(., is.na(.), 0))
str(data_dummy)

# ------------------------ data prep: end

# simple model
#model <- glm(y~.-ID, data=data_dummy)
#summary(model)

# install.packages("mlr3learners", dependencies = TRUE)
library(mlr3)
library("mlr3learners")

# create task, learn and cross validation
creditTask <- TaskClassif$new(id = "data_dummy", backend = data_dummy, target = "y")
learner <- lrn("classif.kknn", id = "dummy", predict_type = "prob")
resampling = rsmp("cv", folds = 3)

# train -> CV
rr = resample(creditTask, learner, resampling, store_models = TRUE)
print(rr)

# evaluate error rate
rr$score(msr("classif.ce"))
rr$aggregate(msr("classif.ce")) # average

# plot resampling result
library("mlr3viz")
library("precrec")

autoplot(rr)
autoplot(rr, type = "roc")

# AUC
rr$score(msr("classif.auc"))
rr$aggregate(msr("classif.auc"))
