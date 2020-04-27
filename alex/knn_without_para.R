# clear all workspace
rm(list = ls())

library(mlr3)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(mlr3learners)
library(data.table)
library(mlr3viz)
library(mlr3tuning)
library(mlr3pipelines)
library(paradox)
library(skimr)
library(gridExtra)

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
setwd("/home/alex/Desktop/i2ml_final_project/")

# suppress package making warning by start up in train 
# Warning: "package ??kknn?? was built under R version 3.6.3"
suppressPackageStartupMessages(library(kknn))

# read data with different encoding
# load data directly into tasks for further training
dl_iv_data <- read.csv2("credit_card_prediction/iv_data/dl_iv_data.csv") %>% mutate(y = as.factor(y))
task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y")

knn_lrn <- lrn("classif.kknn", predict_type = "prob")

knn_param_set0 <- ParamSet$new(params = list(ParamInt$new("k", lower = 5, upper = 100)
))

# k vs. distance
knn_param_set1 <- ParamSet$new(params = list(ParamInt$new("k", lower = 20, upper = 60),
                                            ParamInt$new("distance", lower = 1, upper = 3)
))

# k vs. method
kernel_type = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal")
knn_param_set2 <- ParamSet$new(params = list(ParamInt$new("k", lower = 40, upper = 60),
                                             ParamFct$new("kernel", levels = kernel_type)
))

inner_rsmp <- rsmp("holdout")
knn_auto0 <- AutoTuner$new(learner = knn_lrn, resampling = inner_rsmp, 
                           measures = msr("classif.auc"), tune_ps = knn_param_set0,
                           terminator = term("none"), tuner = tnr("grid_search", resolution = 20))

knn_auto1 <- AutoTuner$new(learner = knn_lrn, resampling = inner_rsmp, 
                          measures = msr("classif.auc"), tune_ps = knn_param_set1,
                          terminator = term("none"), tuner = tnr("grid_search", resolution = 10))

knn_auto2 <- AutoTuner$new(learner = knn_lrn, resampling = inner_rsmp, 
                           measures = msr("classif.auc"), tune_ps = knn_param_set2,
                           terminator = term("none"), tuner = tnr("grid_search", resolution = 10))

# creat a benchmark, knn with 5 fold CV
design <- benchmark_grid(
  tasks = task,
  learners = knn_auto0,
  resampling = rsmp("holdout")
)



# set seed before traing, then run the benchmark
# save the results afterwards
set.seed(2020)
#00:01:21.896 -> 00:06:09.937

# 00:32:17
knn_bmr <- benchmark(design, store_models = TRUE)
knn_results <- knn_bmr$aggregate(measures = msr("classif.auc"))

library(ggplot2)

knn_path0 = knn_bmr$data$learner[[1]]$archive("params")
knn_gg0 = ggplot(knn_path0, aes(
  x = k,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

knn_path1 = knn_bmr$data$learner[[1]]$archive("params")
knn_gg1 = ggplot(knn_path1, aes(
  x = k,
  y = classif.auc, col = factor(distance))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

knn_path2 = knn_bmr$data$learner[[1]]$archive("params")
knn_gg2 = ggplot(knn_path2, aes(
  x = k,
  y = classif.auc, col = factor(kernel))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

# 25-75


# extract confusion matrix for each task
cf_matrix <- function(x) x$prediction()$confusion
knn_result_matrix <- knn_results %>%
  pull(resample_result) %>%
  map(pluck(cf_matrix))
