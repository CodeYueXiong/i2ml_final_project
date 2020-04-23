# clear all workspace
rm(list = ls())
#loading library
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
#set seed
set.seed(2020)
#set work dictionary
setwd("/Users/hebailan/R Programm/i2ml programm/final project/i2ml_final_project/credit_card_prediction")
#load data
dl_iv_data <- read.csv2("./iv_data/dl_iv_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mf_iv_data <- read.csv2("./iv_data/mf_iv_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mice_iv_data <- read.csv2("./iv_data/mice_iv_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))

dl_oh_data <- read.csv("./oh_data/dl_oh_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mf_oh_data <- read.csv("./oh_data/mf_oh_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mice_oh_data <- read.csv("./oh_data/mice_oh_data.csv")  %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))

dl_dm_data <- read.csv2("./dummy_data/dl_dummy_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mf_dm_data <- read.csv2("./dummy_data/mf_dummy_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mice_dm_data <- read.csv2("./dummy_data/mice_dummy_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))




#creat task.
task_without_dm <- list(TaskClassif$new("dl_iv", backend= dl_iv_data , target = "y"),
                        TaskClassif$new("mf_iv", backend= mf_iv_data , target = "y"),
                        TaskClassif$new("mice_iv", backend= mice_iv_data , target = "y"),
                        TaskClassif$new("dl_oh", backend= dl_oh_data , target = "y"),
                        TaskClassif$new("mf_oh", backend= mf_oh_data , target = "y"),
                        TaskClassif$new("mice_oh", backend= mice_oh_data , target = "y")
)


#creat learner
#please change your own name with this format.
xg_learner <- lrn("classif.xgboost", predict_type = "prob")
#set inner_resampling
#warning: if 5 folds cv inner resampling  beyonds your computing power. please reduce it.
inner_rsmp <- rsmp("cv",folds = 5L)
#set measure
all_measures <- msr("classif.auc")


#set param
#please change your own name with this format.
#please set your own parameters.
xg_param_set <- ParamSet$new(
  params = list(ParamInt$new("max_depth", lower = 3, upper = 10),
                ParamDbl$new("min_child_weight", lower = 1, upper = 6),
                # ParamInt$new("max_bin", lower = 500, upper = 1000, default = 900),
                ParamDbl$new("max_delta_step", lower = 1, upper = 6)
                # ParamUty$new("eval_metric", default = "auc")
  )
)

#set terminater
#please check if the diffrent setting of terminaters change your model results.
# all_terminater <- term("combo",
#                        list(term("model_time", secs = 3600),
#                             term("evals", n_evals = 30),
#                             term("stagnation", iters = 5, threshold = 1e-5)))

all_terminater <- term("evals", n_evals = 30)

#set tuning method
all_tuner <-  tnr("random_search")
#creat autotuner
#please change name with the same format
lg_at_learner <- AutoTuner$new(learner = xg_learner, resampling = inner_rsmp, 
                       measures = all_measures, tune_ps = xg_param_set,
                       terminator = all_terminater, tuner = all_tuner)

#set outer_resampling
outer_rsmp <- rsmp("cv", folds = 3L)

#creat a design
#please change name with the same format
design = benchmark_grid(
  tasks = task_without_dm,
  learners = lg_at_learner,
  resamplings = outer_rsmp
)

#set seed
set.seed(2020)
xg_bmr = benchmark(design, store_models = TRUE)
#check the results
#please change name with the same format
xg_results <- xg_bmr$aggregate(all_measures)

# function to extract confusion matrix for each task
cf_matrix <- function(x) x$prediction()$confusion
# extract confusion matrix for each task
xg_result_matrix <- xg_results %>%
  pull(resample_result) %>%
  map(pluck(cf_matrix))

#aut plot results
#please change name with the same format
autoplot(xg_bmr, measure = all_measures) 

#change plot label(if you want)
# + theme(axis.text.x = element_text(angle = 45, hjust = 1))


#autoplot auc for a specific task
#please change name with the same format
autoplot(xg_bmr$clone()$filter(task_id = "mf_iv"), type = "roc")


#plot the path.(to be continue)
