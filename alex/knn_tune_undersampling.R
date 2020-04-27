# clear all workspace
rm(list = ls())

library(mlr3)
library(tidyverse)
library(ggplot2)
library(mlr3learners)
#library(data.table)
#library(mlr3viz)
library(mlr3tuning)
library(mlr3pipelines)
library(paradox)
#library(skimr)
library(smotefamily)
library(gridExtra)

#setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
setwd("/home/alex/Desktop/i2ml_final_project/")

# suppress package making warning by start up in train 
# Warning: "package ??kknn?? was built under R version 3.6.3"
suppressPackageStartupMessages(library(kknn))

# read data with different encoding
# load data directly into tasks for further training
dl_iv_data <- read.csv2("credit_card_prediction/iv_data/dl_iv_data.csv") %>% mutate(y = as.factor(y)) %>% mutate_if(is.integer,as.numeric)
task <- TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y")

# check original class balance
# table(task$truth())

# knn learner
knn_learner <- lrn("classif.kknn", predict_type = "prob")
# po_smote = po("smote", dup_size = 6)
po_under = po("classbalancing",
              id = "undersample", adjust = "major",
              reference = "major", shuffle = FALSE, ratio = 1 / 40)

lrn_under <- GraphLearner$new(po_under %>>% knn_learner, predict_type = "prob")

# setting the tunning for parameters, and terminator
knn_param_set <- ParamSet$new(list(ParamInt$new("classif.kknn.k", lower = 5, upper = 45), 
                                   ParamDbl$new("undersample.ratio", lower = 1/50, upper = 1/40)))

# knn_param_set <- ParamSet$new(params = list(ParamInt$new("classif.kknn.k", lower = 5, upper = 45),
#                                             ParamInt$new("smote.dup_size", lower = 1, upper = 3),
#                                             ParamInt$new("smote.K", lower = 1, upper = 5)
# ))


# knn_param_set$trafo = function(x, param_set) {
#   x$smote.K = round(2^(x$smote.K))
#   x
# }

terms <- term("none")


# creat autotuner, using the inner sampling and tuning parameter with random search
inner_rsmp <- rsmp("cv",folds = 5L)
knn_auto <- AutoTuner$new(learner = lrn_under, resampling = inner_rsmp, 
                          measures = msr("classif.auc"), tune_ps = knn_param_set,
                          terminator = terms, tuner = tnr("grid_search", resolution = 6))

# set outer_resampling, and creat a design with it
outer_rsmp <- rsmp("cv", folds = 3L)
design = benchmark_grid(
  tasks = task,
  learners = knn_auto,
  resamplings = outer_rsmp
)

# 14:08 -> 14:28, 14:34 ->
# set seed before traing, then run the benchmark
# save the results afterwards
set.seed(2020)
knn_bmr <- benchmark(design, store_models = TRUE)
knn_results <- knn_bmr$aggregate(measures = msr("classif.auc"))

# ---------------------------------------------------------------------

# plot color ggplot
library(ggplot2)

under_path1 = knn_bmr$data$learner[[1]]$archive("params")
under_gg1 = ggplot(over_path1, aes(
  x = classif.kknn.k,
  y = classif.auc, col = factor(undersample.ratio))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

under_path2 = knn_bmr$data$learner[[2]]$archive("params")
under_gg2 = ggplot(over_path2, aes(
  x = classif.kknn.k,
  y = classif.auc, col = factor(undersample.ratio))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

under_path3 = knn_bmr$data$learner[[3]]$archive("params")
under_gg3 = ggplot(under_path3, aes(
  x = classif.kknn.k,
  y = classif.auc, col = factor(undersample.ratio))) +
  geom_point(size = 3) +
  geom_line() #+ theme(legend.position = "none")

library(ggpubr)
ggarrange(under_gg1, under_gg2, under_gg3, common.legend = TRUE, legend="bottom")
#grid.arrange(stune_gg1, stune_gg2, stune_gg3, nrow=1)

# ---------------------------------------------------------------------


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
multiplot_roc <- function(models, type="roc"){
  plots <- list()
  model <- models$clone()$filter(task_id = "dl_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[1]] <- autoplot(model, type = type) + ggtitle(paste("dl_iv:", auc))
  
  model <- models$clone()$filter(task_id = "mf_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[2]] <- autoplot(model, type = type) + ggtitle(paste("mf_iv:", auc))
  
  model <- models$clone()$filter(task_id = "mice_iv")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[3]] <- autoplot(model, type = type) + ggtitle(paste("mice_iv:", auc))
  
  model <- models$clone()$filter(task_id = "dl_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[4]] <- autoplot(model, type = type) + ggtitle(paste("dl_oh:", auc))
  
  model <- models$clone()$filter(task_id = "mf_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[5]] <- autoplot(model, type = type) + ggtitle(paste("mf_oh:", auc))
  
  model <- models$clone()$filter(task_id = "mice_oh")
  auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
  plots[[6]] <- autoplot(model, type = type) + ggtitle(paste("mice_oh", auc))
  do.call("grid.arrange", plots)
}

# roc: x= 1-Specificity, y= Sensitivity
# prc: x= Recall, y= Precision

multiplot_roc(knn_bmr)


# KNN performs with no significant difference between different encoding and missing data handling method. Since we used binary variable to idicate whether a category is present or not, the max distance can only be 1 or 0. And other numeric variable have larger distance, meaning that they have a larger impact on the distance then the categorical data, without having significant correlation with our target variable.
# It would be important to either use other training methods, or other ways to handle categorical data better for distance calculation.