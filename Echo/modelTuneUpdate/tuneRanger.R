# clear all workspace
rm(list=ls())


library(tuneRanger)
library(mlr)

#set work dictionary
setwd("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction")
#load data
dl_iv_data <- read.csv2("./iv_data/dl_iv_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mf_iv_data <- read.csv2("./iv_data/mf_iv_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mice_iv_data <- read.csv2("./iv_data/mice_iv_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))

dl_oh_data <- read.csv("./oh_data/dl_oh_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mf_oh_data <- read.csv("./oh_data/mf_oh_data.csv") %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))
mice_oh_data <- read.csv("./oh_data/mice_oh_data.csv")  %>% mutate_if(is.integer,as.numeric) %>% mutate(y= as.factor(y))

task = makeClassifTask(data = dl_iv_data, target = "y")

# Estimate runtime
estimateTimeTuneRanger(task)

set.seed(2020)
# Tuning
res = tuneRanger(task, measure = list(multiclass.brier), num.trees = 1000,
                 num.threads = 2, iters = 70, iters.warmup = 30) 

res
# Ranger Model with the new tuned hyperparameters
res$model