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

# creat a benchmark, knn with 5 fold CV
design <- benchmark_grid(
  tasks = tasks,
  learners = lrn("classif.kknn", predict_type = "prob"),
  resampling = rsmp("cv", folds = 5L)
)

# set seed before traing, then run the benchmark
# save the results afterwards
set.seed(2020)
knn_bmr <- benchmark(design)
knn_results <- knn_bmr$aggregate(measures = msr("classif.auc"))

# extract confusion matrix for each task
cf_matrix <- function(x) x$prediction()$confusion
knn_result_matrix <- knn_results %>%
  pull(resample_result) %>%
  map(pluck(cf_matrix))

# auto plot results
#autoplot(knn_bmr, measure = msr("classif.auc"))

# autoplot auc for all tasks (merged in one plot)
multiplot_roc <- function(models){
  plots <- list()
  plots[[1]] <- autoplot(models$clone()$filter(task_id = "dl_iv"), type = "roc") + xlab("") + ylab("") + ggtitle("dl_iv")
  plots[[2]] <- autoplot(models$clone()$filter(task_id = "mf_iv"), type = "roc") + xlab("") + ylab("") + ggtitle("mf_iv")
  plots[[3]] <- autoplot(models$clone()$filter(task_id = "mice_iv"), type = "roc") + xlab("") + ylab("") + ggtitle("mice_iv")
  plots[[4]] <- autoplot(models$clone()$filter(task_id = "dl_oh"), type = "roc") + xlab("") + ylab("") + ggtitle("dl_oh")
  plots[[5]] <- autoplot(models$clone()$filter(task_id = "mf_oh"), type = "roc") + xlab("") + ylab("") + ggtitle("mf_oh")
  plots[[6]] <- autoplot(models$clone()$filter(task_id = "mice_oh"), type = "roc") + xlab("") + ylab("") + ggtitle("mice_oh")
  do.call("grid.arrange", plots)
}

multiplot_roc(knn_bmr)

