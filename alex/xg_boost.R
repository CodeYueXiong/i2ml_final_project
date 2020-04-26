# clear all workspace
rm(list = ls())

library(mlr3)
library(tidyverse)
library(ggplot2)
library(mlr3learners)
library(data.table)
library(mlr3viz)
library(mlr3tuning)
library(mlr3pipelines)
library(paradox)
library(skimr)
library(smotefamily)
library(gridExtra)

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")

# suppress package making warning by start up in train 
# Warning: "package 'kknn' was built under R version 3.6.3"
# suppressPackageStartupMessages(library(kknn))

# read data with different encoding
dl_iv_data <- read.csv2("credit_card_prediction/iv_data/dl_iv_data.csv") %>% mutate(y = as.factor(y))
# mf_iv_data <- read.csv2("credit_card_prediction/iv_data/mf_iv_data.csv") %>% mutate(y = as.factor(y))
# mice_iv_data <- read.csv2("credit_card_prediction/iv_data/mice_iv_data.csv") %>% mutate(y = as.factor(y))
# dl_oh_data <- read.csv("credit_card_prediction/oh_data/dl_oh_data.csv") %>% mutate(y = as.factor(y))
# mf_oh_data <- read.csv("credit_card_prediction/oh_data/mf_oh_data.csv") %>% mutate(y = as.factor(y))
# mice_oh_data <- read.csv("credit_card_prediction/oh_data/mice_oh_data.csv") %>% mutate(y = as.factor(y))


# load data directly into tasks for further training
# tasks <- list(
#   TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y"),
#   TaskClassif$new("mf_iv", backend = mf_iv_data, target = "y"),
#   TaskClassif$new("mice_iv", backend = mice_iv_data, target = "y"),
#   TaskClassif$new("dl_oh", backend = dl_oh_data, target = "y"),
#   TaskClassif$new("mf_oh", backend = mf_oh_data, target = "y"),
#   TaskClassif$new("mice_oh", backend = mice_oh_data, target = "y")
# )

task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y")

# remove raw data to save memory
# rm(dl_iv_data, mf_iv_data, mice_iv_data, dl_oh_data, mf_oh_data, mice_oh_data)

# xgBoost learner
xg_learner <- lrn("classif.xgboost", predict_type = "prob")

# setting the tunning for parameters, and terminator
xg_smote_param_set <- ParamSet$new(
  params = list(ParamInt$new("max_depth", lower = 3, upper = 15),
                ParamDbl$new("min_child_weight", lower = 5, upper = 10),
                ParamDbl$new("eta", low=0.01, upper=0.3), # learning rate
                #ParamInt$new("gamma", low=0, upper=10),
                ParamDbl$new("subsample", low=0.5, upper=0.8)
                # ParamInt$new("lambda", lower=0, upper=2)
                # ParamInt$new("max_bin", lower = 500, upper = 1000, default = 900),
                # ParamDbl$new("max_delta_step", lower = 1, upper = 6)
                # ParamUty$new("eval_metric", default = "auc")
  )
)

# terms <- term("evals", n_evals = 30)

terms <- term("combo", list(term("model_time", secs = 10),
                            term("evals", n_evals = 100),
                            term("stagnation", iters = 5, threshold = 1e-4)))

# creat autotuner, using the inner sampling and tuning parameter with random search
inner_rsmp <- rsmp("cv",folds = 5L)
xg_auto <- AutoTuner$new(learner = xg_learner, resampling = inner_rsmp, 
                          measures = msr("classif.auc"), tune_ps = xg_smote_param_set,
                          terminator = terms, tuner = tnr("random_search"))

# set outer_resampling, and creat a design with it
outer_rsmp <- rsmp("cv", folds = 3L)
design = benchmark_grid(
  tasks = task,
  learners = xg_auto,
  resamplings = outer_rsmp
)

# set seed before traing, then run the benchmark
# save the results afterwards
set.seed(2020)
xg_bmr <- benchmark(design, store_models = TRUE)
xg_results <- xg_bmr$aggregate(measures = msr("classif.auc"))



# extract confusion matrix for each task
cf_matrix <- function(x) x$prediction()$confusion
xg_result_matrix <- xg_results %>%
  pull(resample_result) %>%
  map(pluck(cf_matrix))

para_results <- xg_bmr$score() %>% 
  pull(learner) %>% 
  map(pluck(c(function(x) x$tuning_result)))

# auto plot results
#autoplot(knn_bmr, measure = msr("classif.auc"))

model <- xg_bmr$clone()$filter(task_id = "dl_iv")
autoplot(model, type = type) + xlab(xlab) + ylab(ylab) + ggtitle(paste("dl_iv:", auc))


# autoplot auc for all tasks (merged in one plot)
# multiplot_roc <- function(models, type="roc", xlab="", ylab=""){
#   plots <- list()
#   model <- models$clone()$filter(task_id = "dl_iv")
#   auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
#   plots[[1]] <- autoplot(model, type = type) + xlab(xlab) + ylab(ylab) + ggtitle(paste("dl_iv:", auc))
#   # plots[[2]] <- autoplot(models$clone()$filter(task_id = "mf_iv"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mf_iv")
#   # plots[[3]] <- autoplot(models$clone()$filter(task_id = "mice_iv"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mice_iv")
#   # plots[[4]] <- autoplot(models$clone()$filter(task_id = "dl_oh"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("dl_oh")
#   # plots[[5]] <- autoplot(models$clone()$filter(task_id = "mf_oh"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mf_oh")
#   # plots[[6]] <- autoplot(models$clone()$filter(task_id = "mice_oh"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mice_oh")
#   do.call("grid.arrange", plots)
# }

# roc: x= 1-Specificity, y= Sensitivity
# prc: x= Recall, y= Precision

# multiplot_roc(xg_bmr)


# KNN performs with no significant difference between different encoding and missing data handling method. Since we used binary variable to idicate whether a category is present or not, the max distance can only be 1 or 0. And other numeric variable have larger distance, meaning that they have a larger impact on the distance then the categorical data, without having significant correlation with our target variable.
# It would be important to either use other training methods, or other ways to handle categorical data better for distance calculation.
