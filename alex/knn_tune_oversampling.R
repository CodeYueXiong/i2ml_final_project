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

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
# setwd("/home/alex/Desktop/i2ml_final_project/")

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

po_over = po("classbalancing",
             id = "oversample", adjust = "minor",
             reference = "minor", shuffle = FALSE, ratio = 6)

lrn_over <- GraphLearner$new(po_over %>>% knn_learner, predict_type = "prob")




# setting the tunning for parameters, and terminator
knn_over_param_set <- ParamSet$new(list(ParamDbl$new("oversample.ratio", lower = 10, upper = 70)))



terms <- term("none")


# creat autotuner, using the inner sampling and tuning parameter with random search
inner_rsmp <- rsmp("cv",folds = 3L)
outer_rsmp <- rsmp("cv", folds = 3L)

knn_over_auto <- AutoTuner$new(learner = lrn_over, resampling = inner_rsmp, 
                               measures = msr("classif.auc"), tune_ps = knn_over_param_set,
                               terminator = terms, tuner = tnr("grid_search", resolution = 5))

# set outer_resampling, and creat a design with it

design_over = benchmark_grid(
  tasks = task,
  learners = knn_over_auto,
  resamplings = outer_rsmp
)



# set seed before traing, then run the benchmark
# save the results afterwards

set.seed(2020)
knn_over_bmr <- benchmark(design_over, store_models = TRUE)

po_smote = po("smote", dup_size = 50)
lrn_smote <- GraphLearner$new(po_smote %>>% knn_learner, predict_type = "prob")
knn_smote_param_set <- ParamSet$new(params = list(ParamInt$new("smote.dup_size", lower = 20, upper = 60),
                                                  ParamInt$new("smote.K", lower = 10, upper = 20)
))
knn_smote_auto <- AutoTuner$new(learner = lrn_smote, resampling = inner_rsmp, 
                                measures = msr("classif.auc"), tune_ps = knn_smote_param_set,
                                terminator = terms, tuner = tnr("grid_search", resolution = 5))

design_smote = benchmark_grid(
  tasks = task,
  learners = knn_smote_auto,
  resamplings = outer_rsmp
)
knn_smote_bmr <- benchmark(design_smote, store_models = TRUE)
# knn_results <- knn_bmr$aggregate(measures = msr("classif.auc"))

# ---------------------------------------------------------------------

# plot color ggplot
library(ggplot2)

over_path1 = knn_over_bmr$data$learner[[1]]$archive("params")
over_gg1 = ggplot(over_path1, aes(
  x = oversample.ratio,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

over_path2 = knn_over_bmr$data$learner[[2]]$archive("params")
over_gg2 = ggplot(over_path2, aes(
  x = oversample.ratio,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

over_path3 = knn_over_bmr$data$learner[[3]]$archive("params")
over_gg3 = ggplot(over_path3, aes(
  x = oversample.ratio,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

library(ggpubr)
ggarrange(over_gg1, over_gg2, over_gg3, common.legend = TRUE, legend="bottom")

over_results <- knn_over_bmr$score() %>%
  pull(learner) %>%
  map(pluck(c(function(x) x$tuning_result)))

# plot color ggplot
smote_path1 = knn_smote_bmr$data$learner[[1]]$archive("params")
smote_gg1 = ggplot(smote_path1, aes(
  x = smote.dup_size,
  y = classif.auc, col = factor(smote.K))) +
  geom_point(size = 3) +
  geom_line() 

smote_path2 = knn_smote_bmr$data$learner[[2]]$archive("params")
smote_gg2 = ggplot(smote_path2, aes(
  x = smote.dup_size,
  y = classif.auc, col = factor(smote.K))) +
  geom_point(size = 3) +
  geom_line()

smote_path3 = knn_smote_bmr$data$learner[[3]]$archive("params")
smote_gg3 = ggplot(smote_path3, aes(
  x = smote.dup_size,
  y = classif.auc, col = factor(smote.K))) +
  geom_point(size = 3) +
  geom_line()

ggarrange(smote_gg1, smote_gg2, smote_gg3, common.legend = TRUE, legend="bottom")

smote_results <- knn_smote_bmr$score() %>%
  pull(learner) %>%
  map(pluck(c(function(x) x$tuning_result)))


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

inner_rsmp <- rsmp("cv",folds = 3L)
outer_rsmp <- rsmp("cv", folds = 3L)

kernel_type <- c("rectangular", "triangular", "epanechnikov", "biweight"
                , "triweight", "cos", "inv", "gaussian", "rank", "optimal")

para_k <- ParamSet$new(params = list(ParamInt$new("classif.kknn.k", lower = 20, upper = 100)))
para_dist <- ParamSet$new(params = list(ParamInt$new("classif.kknn.distance", lower = 1, upper = 5)))
para_kernel <- ParamSet$new(params = list(ParamFct$new("classif.kknn.kernel", levels = kernel_type)))

selected_kern <- c("triangular", "biweight", "epanechnikov")
knn_over_lrn <- GraphLearner$new(po_over_tuned %>>% lrn("classif.kknn", predict_type = "prob", distance=1), predict_type = "prob")
para_k <- ParamSet$new(params = list(ParamInt$new("classif.kknn.k", lower = 20, upper = 100), 
                                     ParamFct$new("classif.kknn.kernel", levels = selected_kern)))

knn_over_k <- AutoTuner$new(
  learner = knn_over_lrn, resampling = inner_rsmp,
  measures = msr("classif.auc"), tune_ps = para_k,
  terminator = term("none"), tuner = tnr("grid_search", resolution = 5))

design_over_k = benchmark_grid(
  tasks = task,
  learners = knn_over_k,
  resamplings = outer_rsmp
)

bmr_over_k <- run_benchmark(design_over_k)


# ----------- benchmark

library(tictoc)
run_benchmark <- function(design){
  set.seed(2020)
  bmr <- benchmark(design, store_models = TRUE)
  run_benchmark <- bmr
}

# oversampling

# knn_bmr$score() %>% 
#     pull(learner) %>%
#     map(pluck(c(function(x) x$tuning_result)))

po_over_tuned <- po("classbalancing",
                    id = "oversample", adjust = "minor",
                    reference = "minor", shuffle = FALSE, ratio = 25)
knn_over_lrn <- GraphLearner$new(po_over_tuned %>>% lrn("classif.kknn", predict_type = "prob", distance=1), predict_type = "prob")


knn_over_dist <- AutoTuner$new(
  learner = knn_over_lrn, resampling = inner_rsmp,
  measures = msr("classif.auc"), tune_ps = para_dist,
  terminator = term("none"), tuner = tnr("grid_search", resolution = 5))

design_over_dist <- benchmark_grid(
  tasks = task,
  learners = knn_over_dist,
  resamplings = outer_rsmp
)

bmr_over_dist <- run_benchmark(design_over_dist)
over_dist_para <- bmr_over_dist$score() %>%
    pull(learner) %>%
    map(pluck(c(function(x) x$tuning_result)))



knn_over_lrn <- GraphLearner$new(po_over_tuned %>>% lrn("classif.kknn", predict_type = "prob", distance=1), predict_type = "prob")

knn_over_kernel <- AutoTuner$new(
  learner = knn_over_lrn, resampling = inner_rsmp,
  measures = msr("classif.auc"), tune_ps = para_kernel,
  terminator = term("none"), tuner = tnr("grid_search", resolution = 5))

design_over_kernel <- benchmark_grid(
  tasks = task,
  learners = knn_over_kernel,
  resamplings = outer_rsmp
)

bmr_over_kern <- run_benchmark(design_over_kernel)
over_kern_para <- bmr_over_kern$score() %>%
  pull(learner) %>%
  map(pluck(c(function(x) x$tuning_result)))


knn_over_lrn <- GraphLearner$new(po_over_tuned %>>% lrn("classif.kknn", predict_type = "prob", distance=1, kernel="biweight"), predict_type = "prob")
knn_over_k <- AutoTuner$new(
  learner = knn_over_lrn, resampling = inner_rsmp,
  measures = msr("classif.auc"), tune_ps = para_k,
  terminator = term("none"), tuner = tnr("grid_search", resolution = 5))

design_over_k = benchmark_grid(
  tasks = task,
  learners = knn_over_k,
  resamplings = outer_rsmp
)

bmr_over_k <- run_benchmark(design_over_k)



# smote

po_smote_tuned <- po("smote", dup_size = 40, K = 15)
knn_smote_lrn <- GraphLearner$new(po_smote_tuned %>>% lrn("classif.kknn", predict_type = "prob"), predict_type = "prob")


knn_smote_dist <- AutoTuner$new(
  learner = knn_smote_lrn, resampling = inner_rsmp,
  measures = msr("classif.auc"), tune_ps = para_dist,
  terminator = term("none"), tuner = tnr("grid_search", resolution = 5))

design_smote_dist <- benchmark_grid(
  tasks = task,
  learners = knn_smote_dist,
  resamplings = outer_rsmp
)

bmr_smote_dist <- run_benchmark(design_smote_dist)


knn_smote_lrn <- GraphLearner$new(po_smote_tuned %>>% lrn("classif.kknn", predict_type = "prob", distance=3), predict_type = "prob")
knn_smote_kernel <- AutoTuner$new(
  learner = knn_smote_lrn, resampling = inner_rsmp,
  measures = msr("classif.auc"), tune_ps = para_kernel,
  terminator = term("none"), tuner = tnr("grid_search", resolution = 10))

design_smote_kernel <- benchmark_grid(
  tasks = task,
  learners = knn_smote_kernel,
  resamplings = outer_rsmp
)

bmr_smote_kern <- run_benchmark(design_smote_kernel)

selected_kern <- c("rectangular", "triangular", "gaussian")
selected_kern_over <- c("triangular", "biweight", "epanechnikov")
para_k <- ParamSet$new(params = list(ParamInt$new("classif.kknn.k", lower = 20, upper = 100), 
                                     ParamFct$new("classif.kknn.kernel", levels = selected_kern)))

knn_smote_lrn <- GraphLearner$new(po_smote_tuned %>>% lrn("classif.kknn", predict_type = "prob", distance=3), predict_type = "prob")
knn_smote_k <- AutoTuner$new(
  learner = knn_smote_lrn, resampling = inner_rsmp,
  measures = msr("classif.auc"), tune_ps = para_k,
  terminator = term("none"), tuner = tnr("grid_search", resolution = 5))

design_smote_k <- benchmark_grid(
  
  tasks = task,
  learners = knn_smote_k,
  resamplings = outer_rsmp
)

bmr_smote_k <- run_benchmark(design_smote_k)
# 16:24:31 -> 18:32:37

# ----------------------- plot

# --- over

# k

para_results <- bmr_over_k$score() %>% 
  pull(learner) %>%
  map(pluck(c(function(x) x$tuning_result)))

over_k_path1 = bmr_over_k$data$learner[[1]]$archive("params")
over_k_gg1 = ggplot(over_k_path1, aes(
  x = classif.kknn.k,
  y = classif.auc, col = classif.kknn.kernel)) +
  geom_point(size = 3) +
  geom_line()

over_k_path2 = bmr_over_k$data$learner[[2]]$archive("params")
over_k_gg2 = ggplot(over_k_path2, aes(
  x = classif.kknn.k,
  y = classif.auc, col = classif.kknn.kernel)) +
  geom_point(size = 3) +
  geom_line()

over_k_path3 = bmr_over_k$data$learner[[3]]$archive("params")
over_k_gg3 = ggplot(over_k_path3, aes(
  x = classif.kknn.k,
  y = classif.auc, col = classif.kknn.kernel)) +
  geom_point(size = 3) +
  geom_line()

ggarrange(over_k_gg1, over_k_gg2, over_k_gg3, common.legend = TRUE, legend="bottom")

# dist

over_dist_path1 = bmr_over_dist$data$learner[[1]]$archive("params")
over_dist_gg1 = ggplot(over_dist_path1, aes(
  x = classif.kknn.distance,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

over_dist_path2 = bmr_over_dist$data$learner[[2]]$archive("params")
over_dist_gg2 = ggplot(over_dist_path2, aes(
  x = classif.kknn.distance,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

over_dist_path3 = bmr_over_dist$data$learner[[3]]$archive("params")
over_dist_gg3 = ggplot(over_dist_path3, aes(
  x = classif.kknn.distance,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

ggarrange(over_dist_gg1, over_dist_gg2, over_dist_gg3, common.legend = TRUE, legend="bottom")

# kernel

thm = theme(axis.text.x = element_text(angle = 45, hjust = 1))
over_kern_path1 = bmr_over_kern$data$learner[[1]]$archive("params")
over_kern_gg1 = ggplot(over_kern_path1, aes(
  x = classif.kknn.kernel,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() + thm

over_kern_path2 = bmr_over_kern$data$learner[[2]]$archive("params")
over_kern_gg2 = ggplot(over_kern_path2, aes(
  x = classif.kknn.kernel,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() + thm

over_kern_path3 = bmr_over_kern$data$learner[[3]]$archive("params")
over_kern_gg3 = ggplot(over_kern_path3, aes(
  x = classif.kknn.kernel,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() + thm

ggarrange(over_kern_gg1, over_kern_gg2, over_kern_gg3, common.legend = TRUE, legend="bottom")

para_results <- bmr_over_kern$score() %>% 
    pull(learner) %>%
    map(pluck(c(function(x) x$tuning_result)))


# --- smote

# k

smote_k_path1 = bmr_smote_k$data$learner[[1]]$archive("params")
smote_k_gg1 = ggplot(smote_k_path1, aes(
  x = classif.kknn.k,
  y = classif.auc, col = classif.kknn.kernel)) +
  geom_point(size = 3) +
  geom_line()

smote_k_path2 = bmr_smote_k$data$learner[[2]]$archive("params")
smote_k_gg2 = ggplot(smote_k_path2, aes(
  x = classif.kknn.k,
  y = classif.auc, col = classif.kknn.kernel)) +
  geom_point(size = 3) +
  geom_line()

smote_k_path3 = bmr_smote_k$data$learner[[3]]$archive("params")
smote_k_gg3 = ggplot(smote_k_path3, aes(
  x = classif.kknn.k,
  y = classif.auc, col = classif.kknn.kernel)) +
  geom_point(size = 3) +
  geom_line()

ggarrange(smote_k_gg1, smote_k_gg2, smote_k_gg3, common.legend = TRUE, legend="bottom")

# dist

smote_dist_path1 = bmr_smote_dist$data$learner[[1]]$archive("params")
smote_dist_gg1 = ggplot(smote_dist_path1, aes(
  x = classif.kknn.distance,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

smote_dist_path2 = bmr_smote_dist$data$learner[[2]]$archive("params")
smote_dist_gg2 = ggplot(smote_dist_path2, aes(
  x = classif.kknn.distance,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

smote_dist_path3 = bmr_smote_dist$data$learner[[3]]$archive("params")
smote_dist_gg3 = ggplot(smote_dist_path3, aes(
  x = classif.kknn.distance,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line()

ggarrange(smote_dist_gg1, smote_dist_gg2, smote_dist_gg3, common.legend = TRUE, legend="bottom")

# kernel

thm = theme(axis.text.x = element_text(angle = 45, hjust = 1))
smote_kern_path1 = bmr_smote_kern$data$learner[[1]]$archive("params")
smote_kern_gg1 = ggplot(smote_kern_path1, aes(
  x = classif.kknn.kernel,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() + thm

smote_kern_path2 = bmr_smote_kern$data$learner[[2]]$archive("params")
smote_kern_gg2 = ggplot(smote_kern_path2, aes(
  x = classif.kknn.kernel,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() + thm

smote_kern_path3 = bmr_smote_kern$data$learner[[3]]$archive("params")
smote_kern_gg3 = ggplot(smote_kern_path3, aes(
  x = classif.kknn.kernel,
  y = classif.auc)) +
  geom_point(size = 3) +
  geom_line() + thm

ggarrange(smote_kern_gg1, smote_kern_gg2, smote_kern_gg3, common.legend = TRUE, legend="bottom")


# 
# # ---------------------------------------------------------------------
# 
# 
# # --------- old iv
# # nr  resample_result task_id         learner_id resampling_id iters classif.auc
# # 1:  1 <ResampleResult>   dl_iv classif.kknn.tuned            cv     3   0.6919682
# # 2:  2 <ResampleResult>   mf_iv classif.kknn.tuned            cv     3   0.6760526
# # 3:  3 <ResampleResult> mice_iv classif.kknn.tuned            cv     3   0.6738909
# # 4:  4 <ResampleResult>   dl_oh classif.kknn.tuned            cv     3   0.7044445
# # 5:  5 <ResampleResult>   mf_oh classif.kknn.tuned            cv     3   0.6749367
# # 6:  6 <ResampleResult> mice_oh classif.kknn.tuned            cv     3   0.6818601
# 
# # --------- new iv: n_evals
# 
# 
# 
# # extract confusion matrix for each task
# cf_matrix <- function(x) x$prediction()$confusion
# knn_result_matrix <- knn_results %>%
#   pull(resample_result) %>%
#   map(pluck(cf_matrix))
# 
# para_results <- knn_bmr$score() %>% 
#   pull(learner) %>% 
#   map(pluck(c(function(x) x$tuning_result)))
# 
# # auto plot results
# #autoplot(knn_bmr, measure = msr("classif.auc"))
# 
# 
# # autoplot auc for all tasks (merged in one plot)
# multiplot_roc <- function(models, type="roc"){
#   plots <- list()
#   model <- models$clone()$filter(task_id = "dl_iv")
#   auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
#   plots[[1]] <- autoplot(model, type = type) + ggtitle(paste("dl_iv:", auc))
#   
#   model <- models$clone()$filter(task_id = "mf_iv")
#   auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
#   plots[[2]] <- autoplot(model, type = type) + ggtitle(paste("mf_iv:", auc))
#   
#   model <- models$clone()$filter(task_id = "mice_iv")
#   auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
#   plots[[3]] <- autoplot(model, type = type) + ggtitle(paste("mice_iv:", auc))
#   
#   model <- models$clone()$filter(task_id = "dl_oh")
#   auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
#   plots[[4]] <- autoplot(model, type = type) + ggtitle(paste("dl_oh:", auc))
#   
#   model <- models$clone()$filter(task_id = "mf_oh")
#   auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
#   plots[[5]] <- autoplot(model, type = type) + ggtitle(paste("mf_oh:", auc))
#   
#   model <- models$clone()$filter(task_id = "mice_oh")
#   auc <- round(model$aggregate(msr("classif.auc"))[[7]], 4)
#   plots[[6]] <- autoplot(model, type = type) + ggtitle(paste("mice_oh", auc))
#   do.call("grid.arrange", plots)
# }
# 
# # roc: x= 1-Specificity, y= Sensitivity
# # prc: x= Recall, y= Precision
# 
# multiplot_roc(knn_bmr)


# KNN performs with no significant difference between different encoding and missing data handling method. Since we used binary variable to idicate whether a category is present or not, the max distance can only be 1 or 0. And other numeric variable have larger distance, meaning that they have a larger impact on the distance then the categorical data, without having significant correlation with our target variable.
# It would be important to either use other training methods, or other ways to handle categorical data better for distance calculation.