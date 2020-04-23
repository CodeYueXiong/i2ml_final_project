# clear all workspace
rm(list = ls())
# loading library
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
# set seed
set.seed(2020)
# set work dictionary
setwd("/Users/hebailan/R Programm/i2ml programm/final project/i2ml_final_project/credit_card_prediction")
# read data
dl_iv_data <- read.csv2("./iv_data/dl_iv_data.csv") %>% mutate(y = as.factor(y))
mf_iv_data <- read.csv2("./iv_data/mf_iv_data.csv") %>% mutate(y = as.factor(y))
mice_iv_data <- read.csv2("./iv_data/mice_iv_data.csv") %>% mutate(y = as.factor(y))
dl_oh_data <- read.csv("./oh_data/dl_oh_data.csv") %>% mutate(y = as.factor(y))
mf_oh_data <- read.csv("./oh_data/mf_oh_data.csv") %>% mutate(y = as.factor(y))
mice_oh_data <- read.csv("./oh_data/mice_oh_data.csv") %>% mutate(y = as.factor(y))


# creat task for all the data
task_all <- list(
  TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y"),
  TaskClassif$new("mf_iv", backend = mf_iv_data, target = "y"),
  TaskClassif$new("mice_iv", backend = mice_iv_data, target = "y"),
  TaskClassif$new("dl_oh", backend = dl_oh_data, target = "y"),
  TaskClassif$new("mf_oh", backend = mf_oh_data, target = "y"),
  TaskClassif$new("mice_oh", backend = mice_oh_data, target = "y"),
  TaskClassif$new("dl_dm", backend = dl_dm_data, target = "y"),
  TaskClassif$new("mf_dm", backend = mf_dm_data, target = "y"),
  TaskClassif$new("mice_dm", backend = mice_dm_data, target = "y")
)

# creat task without dm
task_without_dm <- list(
  TaskClassif$new("dl_iv", backend = dl_iv_data, target = "y"),
  TaskClassif$new("mf_iv", backend = mf_iv_data, target = "y"),
  TaskClassif$new("mice_iv", backend = mice_iv_data, target = "y"),
  TaskClassif$new("dl_oh", backend = dl_oh_data, target = "y"),
  TaskClassif$new("mf_oh", backend = mf_oh_data, target = "y"),
  TaskClassif$new("mice_oh", backend = mice_oh_data, target = "y")
)

# creat a benchmark
design <- benchmark_grid(
  tasks = task_all,
  learners = lrn("classif.log_reg", predict_type = "prob"),
  resampling = rsmp("cv", folds = 5L)
)
# set measure
all_measures <- msr("classif.auc")

# run the benchmark
lg_bmr <- benchmark(design)
# save the results
lg_results <- lg_bmr$aggregate(measures = all_measures)

# function to extract confusion matrix for each task
cf_matrix <- function(x) x$prediction()$confusion
# extract confusion matrix for each task
lg_result_matrix <- lg_results %>%
  pull(resample_result) %>%
  map(pluck(cf_matrix))

# aut plot results
# please change name with the same format
autoplot(lg_bmr, measure = all_measures)

# change plot label(if you want)
# + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# autoplot auc for a specific task
# please change name with the same format
autoplot(lg_bmr$clone()$filter(task_id = "mf_iv"), type = "roc")
