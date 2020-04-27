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

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")

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

# dl_oh
data <- tasks[[4]]$data()
task <- tasks[[4]]

skimr::skim(data)

# check original class balance
table(tasks[[1]]$truth())


# undersample majority class (relative to majority class)
po_under = po("classbalancing",
              id = "undersample", adjust = "major",
              reference = "major", shuffle = FALSE, ratio = 1 / 25)
# reduce majority class by factor '1/ratio'
table(po_under$train(list(task))$output$truth())


# oversample majority class (relative to majority class)
po_over = po("classbalancing",
             id = "oversample", adjust = "minor",
             reference = "minor", shuffle = FALSE, ratio = 25)
# enrich minority class by factor 'ratio'
table(po_over$train(list(task))$output$truth())


# SMOTE enriches the minority class with synthetic data (Synthetic Minority Oversampling Technique)
po_smote = po("smote", dup_size = 6)
# enrich minority class by factor (dup_size + 1)
table(po_smote$train(list(task))$output$truth())

# -------------------------------------
# -------------------------------------
# -------------------------------------

lrn = lrn("classif.kknn")

# combine learner with pipeline graph
lrn_under = GraphLearner$new(po_under %>>% lrn)
lrn_over = GraphLearner$new(po_over %>>% lrn)
# lrn_smote = GraphLearner$new(po_smote %>>% lrn)

ps_under = ParamSet$new(list(ParamDbl$new("undersample.ratio", lower = 1 / 6, upper = 1)))
ps_over = ParamSet$new(list(ParamDbl$new("oversample.ratio", lower = 1, upper = 6)))
# ps_smote = ParamSet$new(list(
#   ParamInt$new("smote.dup_size", lower = 1, upper = 6),
#   ParamInt$new("smote.K", lower = 1, upper = 6)
# ))

