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



task_all <- TaskClassif$new("dl_iv", backend = mf_iv_data, target = "y")

# creat a benchmark
design <- benchmark_grid(
  tasks = task_without_dm,
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






#creat smote learner:
po_smote = po("smote", dup_size = 60)
# enrich minority class by factor (dup_size + 1)
table(po_smote$train(list(task_all)$output$truth()))
#learner
learner = lrn("classif.log_reg", predict_type = "prob")
lg_smote_lrn = GraphLearner$new(po_smote %>>% learner, predict_type = "prob")
#set inner_resampling
#warning: if 5 folds cv inner resampling  beyonds your computing power. please reduce it.
inner_rsmp <- rsmp("cv",folds = 5L)
#set measure
all_measures <- msr("classif.auc")
#set param
#please change your own name with this format.
#please set your own parameters.
lg_smote_param <- ParamSet$new(
  params = list(ParamInt$new("smote.dup_size", lower = 40, upper = 60),
                ParamInt$new("smote.K", lower = 1, upper = 6)
  )
)

#set terminater
#please check if the diffrent setting of terminaters change your model results.
# all_terminater <- term("combo",
#                        list(term("model_time", secs = 3600),
#                             term("evals", n_evals = 30),
#                             term("stagnation", iters = 5, threshold = 1e-5)))

all_terminater <- term("evals", n_evals = 5)

#set tuning method
all_tuner <-  tnr("random_search")
#creat autotuner
#please change name with the same format
lg_at_learner <- AutoTuner$new(learner = lg_smote_lrn, resampling = inner_rsmp, 
                               measures = all_measures, tune_ps = lg_smote_param,
                               terminator = all_terminater, tuner = all_tuner)

#set outer_resampling
outer_rsmp <- rsmp("cv", folds = 3L)

#creat a design
#please change name with the same format
smote_design = benchmark_grid(
  tasks = task_without_dm,
  learners = lg_at_learner,
  resamplings = outer_rsmp
)

#set seed
set.seed(2020)
lg_smote_bmr = benchmark(smote_design, store_models = TRUE)


  
# save the results
lg_smote_results <- lg_smote_bmr$aggregate(measures = all_measures)


perform <- lg_smote_bmr$score()


para_results <- lg_smote_bmr$score() %>% 
  pull(learner) %>% 
  map(pluck(c(function(x) x$tuning_result)))




# function to extract confusion matrix for each task
cf_matrix <- function(x) x$prediction()$confusion
# extract confusion matrix for each task
lg_smote_result_matrix <- lg_smote_results %>%
  pull(resample_result) %>%
  map(pluck(cf_matrix))

# aut plot results
# please change name with the same format
autoplot(lg_smote_bmr, measure = all_measures)

# change plot label(if you want)
# + theme(axis.text.x = element_text(angle = 45, hjust = 1))

# autoplot auc for a specific task
# please change name with the same format
autoplot(lg_smote_bmr$clone()$filter(task_id = "mf_iv"), type = "roc")