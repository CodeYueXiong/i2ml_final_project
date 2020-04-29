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

library(tictoc) # time measurement

#setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
setwd("/home/alex/Desktop/i2ml_final_project/")

# suppress package making warning by start up in train 
# Warning: "package ??kknn?? was built under R version 3.6.3"
suppressPackageStartupMessages(library(kknn))

# read data with different encoding
# load data directly into tasks for further training
dl_iv_data <- read.csv2("credit_card_prediction/iv_data/dl_iv_data.csv") # %>% mutate(y = as.factor(y))
dl_iv_data$y <- as.factor(dl_iv_data$y)

task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y")

# knn with 3 different paramSets, to analysis how to do further tuning
knn_lrn <- lrn("classif.kknn", predict_type = "prob")

# only k
param_k <- ParamSet$new(params = list(ParamInt$new("k", lower = 5, upper = 100)))

# k vs. distance
param_k_dist <- ParamSet$new(params = list(ParamInt$new("k", lower = 20, upper = 100),
                                            ParamInt$new("distance", lower = 1, upper = 3)))

knn_lrn <- lrn("classif.kknn", predict_type = "prob", distance = 1)
# k vs. kernel
kernel_type = c("rectangular", "triangular", "epanechnikov", "biweight", "triweight", "cos", "inv", "gaussian", "rank", "optimal")
param_k_kernel <- ParamSet$new(params = list(ParamInt$new("k", lower = 40, upper = 100),
                                             ParamFct$new("kernel", levels = kernel_type)))


param_all <- ParamSet$new(params = list(ParamInt$new("k", lower = 40, upper = 100),
                                        ParamInt$new("distance", lower = 1, upper = 3),
                                             ParamFct$new("kernel", levels = kernel_type)))

inner_rsmp <- rsmp("holdout")

# create three AutoTuner with different paramSets
knn_auto_k <- AutoTuner$new(learner = knn_lrn, resampling = inner_rsmp, 
                           measures = msr("classif.auc"), tune_ps = param_k,
                           terminator = term("none"), tuner = tnr("grid_search", resolution = 30))

knn_auto_dist <- AutoTuner$new(learner = knn_lrn, resampling = inner_rsmp, 
                          measures = msr("classif.auc"), tune_ps = param_k_dist,
                          terminator = term("none"), tuner = tnr("grid_search", resolution = 10))

knn_auto_kern <- AutoTuner$new(learner = knn_lrn, resampling = inner_rsmp, 
                           measures = msr("classif.auc"), tune_ps = param_k_kernel,
                           terminator = term("none"), tuner = tnr("grid_search", resolution = 10))

# outer
outer_rsmp <- rsmp("holdout")

design_k <- benchmark_grid(
  tasks = task,
  learners = knn_auto_k,
  resampling = outer_rsmp 
)

design_dist <- benchmark_grid(
  tasks = task,
  learners = knn_auto_dist,
  resampling = outer_rsmp
)

design_kern <- benchmark_grid(
  tasks = task,
  learners = knn_auto_kern,
  resampling = outer_rsmp
)


run_benchmark <- function(design){
  set.seed(2020)
  tic()
  bmr <- benchmark(design, store_models = TRUE)
  toc()
  run_benchmark <- bmr
}


bmr_k <- run_benchmark(design_k)
bmr_dist <- run_benchmark(design_dist)
bmr_kern <- run_benchmark(design_kern)

#knn_results <- knn_bmr$aggregate(measures = msr("classif.auc"))

library(ggplot2)

k_path = bmr_k$data$learner[[1]]$archive("params")
knn_ggp1 = ggplot(k_path, aes(
  x = k,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

dist_path = bmr_dist$data$learner[[1]]$archive("params")
knn_ggp2 = ggplot(dist_path, aes(
  x = k,
  y = classif.auc, col = factor(distance))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

kern_path = bmr_kern$data$learner[[1]]$archive("params")
knn_ggp3 = ggplot(kern_path, aes(
  x = k,
  y = classif.auc, col = factor(kernel))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")


knn_ggp1
knn_ggp2
knn_ggp3

# fix distance to '1', and kernel to 'inv'
knn_lrn2 <- lrn("classif.kknn", predict_type = "prob", distance = 1, kernel = "inv")
param_k2 <- ParamSet$new(params = list(ParamInt$new("k", lower = 5, upper = 100)))
knn_auto_k2 <- AutoTuner$new(learner = knn_lrn2, resampling = inner_rsmp, 
                            measures = msr("classif.auc"), tune_ps = param_k2,
                            terminator = term("none"), tuner = tnr("grid_search", resolution = 30))

design_k2 <- benchmark_grid(
  tasks = task,
  learners = knn_auto_k2,
  resampling = outer_rsmp 
)
bmr_k2 <- run_benchmark(design_k2)

k_path2 = bmr_k2$data$learner[[1]]$archive("params")
knn_ggp4 = ggplot(k_path2, aes(
  x = k,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")


# -------------------------------------

knn_auto_all <- AutoTuner$new(learner = knn_lrn, resampling = inner_rsmp, 
                               measures = msr("classif.auc"), tune_ps = param_all,
                               terminator = term("none"), tuner = tnr("grid_search", resolution = 10))

design_all <- benchmark_grid(
  tasks = task,
  learners = knn_auto_all,
  resampling = outer_rsmp
)

bmr_all <- run_benchmark(design_all)

k_path_all = bmr_all$data$learner[[1]]$archive("params")
knn_ggp_all = ggplot(k_path_all, aes(
  x = k,
  y = classif.auc, col=factor(kernel), linetype=factor(distance))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")
