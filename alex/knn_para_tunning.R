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
# Warning: "package ¡¥kknn¡¦ was built under R version 3.6.3"
suppressPackageStartupMessages(library(kknn))

# read data with different encoding
dl_iv_data <- read.csv2("credit_card_prediction/iv_data/dl_iv_data.csv") %>% mutate(y = as.factor(y))
mf_iv_data <- read.csv2("credit_card_prediction/iv_data/mf_iv_data.csv") %>% mutate(y = as.factor(y))
mice_iv_data <- read.csv2("credit_card_prediction/iv_data/mice_iv_data.csv") %>% mutate(y = as.factor(y))
dl_oh_data <- read.csv("credit_card_prediction/oh_data/dl_oh_data.csv") %>% mutate(y = as.factor(y))
mf_oh_data <- read.csv("credit_card_prediction/oh_data/mf_oh_data.csv") %>% mutate(y = as.factor(y))
mice_oh_data <- read.csv("credit_card_prediction/oh_data/mice_oh_data.csv") %>% mutate(y = as.factor(y))


# load data directly into tasks for further training
tasks <- list(
  TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y"),
  TaskClassif$new("mf_iv", backend = mf_iv_data, target = "y"),
  TaskClassif$new("mice_iv", backend = mice_iv_data, target = "y"),
  TaskClassif$new("dl_oh", backend = dl_oh_data, target = "y"),
  TaskClassif$new("mf_oh", backend = mf_oh_data, target = "y"),
  TaskClassif$new("mice_oh", backend = mice_oh_data, target = "y")
)

# remove raw data to save memory
rm(dl_iv_data, mf_iv_data, mice_iv_data, dl_oh_data, mf_oh_data, mice_oh_data)

# knn learner
knn_learner <- lrn("classif.kknn", predict_type = "prob")

# setting the tunning for parameters, and terminator
knn_param_set <- ParamSet$new(params = list(ParamInt$new("k", lower = 1, upper = 50)))
terms <- term("combo", list(term("model_time", secs = 360),
                           term("evals", n_evals = 100),
                           term("stagnation", iters = 5, threshold = 1e-4)))


# creat autotuner, using the inner sampling and tuning parameter with random search
inner_rsmp <- rsmp("cv",folds = 5L)
knn_auto <- AutoTuner$new(learner = knn_learner, resampling = inner_rsmp, 
                               measures = msr("classif.auc"), tune_ps = knn_param_set,
                               terminator = terms, tuner = tnr("random_search"))

# set outer_resampling, and creat a design with it
outer_rsmp <- rsmp("cv", folds = 3L)
design = benchmark_grid(
  tasks = tasks,
  learners = knn_auto,
  resamplings = outer_rsmp
)

# set seed before traing, then run the benchmark
# save the results afterwards
set.seed(2020)
knn_bmr <- benchmark(design, store_models = TRUE)
knn_results <- knn_bmr$aggregate(measures = msr("classif.auc"))

# --------- old iv
# nr  resample_result task_id         learner_id resampling_id iters classif.auc
# 1:  1 <ResampleResult>   dl_iv classif.kknn.tuned            cv     3   0.6919682
# 2:  2 <ResampleResult>   mf_iv classif.kknn.tuned            cv     3   0.6760526
# 3:  3 <ResampleResult> mice_iv classif.kknn.tuned            cv     3   0.6738909
# 4:  4 <ResampleResult>   dl_oh classif.kknn.tuned            cv     3   0.7044445
# 5:  5 <ResampleResult>   mf_oh classif.kknn.tuned            cv     3   0.6749367
# 6:  6 <ResampleResult> mice_oh classif.kknn.tuned            cv     3   0.6818601

# --------- new iv: n_evals



# extract confusion matrix for each task
cf_matrix <- function(x) x$prediction()$confusion
knn_result_matrix <- knn_results %>%
  pull(resample_result) %>%
  map(pluck(cf_matrix))

para_results <- knn_bmr$score() %>% 
  pull(learner) %>% 
  map(pluck(c(function(x) x$tuning_result)))

# auto plot results
#autoplot(knn_bmr, measure = msr("classif.auc"))


# autoplot auc for all tasks (merged in one plot)
multiplot_roc <- function(models, type="roc", xlab="", ylab=""){
  plots <- list()
  plots[[1]] <- autoplot(models$clone()$filter(task_id = "dl_iv"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("dl_iv")
  plots[[2]] <- autoplot(models$clone()$filter(task_id = "mf_iv"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mf_iv")
  plots[[3]] <- autoplot(models$clone()$filter(task_id = "mice_iv"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mice_iv")
  plots[[4]] <- autoplot(models$clone()$filter(task_id = "dl_oh"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("dl_oh")
  plots[[5]] <- autoplot(models$clone()$filter(task_id = "mf_oh"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mf_oh")
  plots[[6]] <- autoplot(models$clone()$filter(task_id = "mice_oh"), type = type) + xlab(xlab) + ylab(ylab) + ggtitle("mice_oh")
  do.call("grid.arrange", plots)
}

# roc: x= 1-Specificity, y= Sensitivity
# prc: x= Recall, y= Precision

multiplot_roc(knn_bmr)


# KNN performs with no significant difference between different encoding and missing data handling method. Since we used binary variable to idicate whether a category is present or not, the max distance can only be 1 or 0. And other numeric variable have larger distance, meaning that they have a larger impact on the distance then the categorical data, without having significant correlation with our target variable.
# It would be important to either use other training methods, or other ways to handle categorical data better for distance calculation.