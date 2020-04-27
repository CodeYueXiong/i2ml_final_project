# read data
library(mlr3)
library(janitor)

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")

dataToTask <- function(path, id, sep=';', header=TRUE){
  dt <- read.csv2(path, sep = sep, header = header)
  dt <- as.data.frame(sapply(dt, as.numeric))
  dt$y <- as.factor(dt$y)
  dt <- clean_names(dt)
  dataToTask <- TaskClassif$new(id = id, backend = dt, target = "y")
}

train_model <- function(task, learner, resampling){
  set.seed(2020)
  print(task$id)
  train_model <- resample(task, learner, resampling, store_models = TRUE)
  print(train_model)
}

# dl_oh_task <- dataToTask("credit_card_prediction/oh_data/dl_oh_data.csv", "dl_oh", sep=',')
# mf_oh_task <- dataToTask("credit_card_prediction/oh_data/mf_oh_data.csv", "mf_oh", sep=',')
# mice_oh_task <- dataToTask("credit_card_prediction/oh_data/mice_oh_data.csv", "mice_oh", sep=',')

old_dl_iv_task <- dataToTask("credit_card_prediction/iv_data/old_dl_iv.csv", "old_dl_iv")
old_mf_iv_task <- dataToTask("credit_card_prediction/iv_data/old_mf_iv.csv", "old_mf_iv")
old_mice_iv_task <- dataToTask("credit_card_prediction/iv_data/old_mice_iv.csv", "old_mice_iv")

dl_iv_task <- dataToTask("credit_card_prediction/iv_data/dl_iv_data.csv", "dl_iv")
mf_iv_task <- dataToTask("credit_card_prediction/iv_data/mf_iv_data.csv", "mf_iv")
mice_iv_task <- dataToTask("credit_card_prediction/iv_data/mice_iv_data.csv", "mice_iv")

data_old <- old_dl_iv_task$data()
data_new <- dl_iv_task$data()


resampling = rsmp("cv", folds = 5)
learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=2, scale=FALSE)

models <- list()
models[[1]] <- train_model(old_dl_iv_task, learner, resampling)
models[[2]] <- train_model(dl_iv_task, learner, resampling)
models[[3]] <- train_model(old_mf_iv_task, learner, resampling)
models[[4]] <- train_model(mf_iv_task, learner, resampling)
models[[5]] <- train_model(old_mice_iv_task, learner, resampling)
models[[6]] <- train_model(mice_iv_task, learner, resampling)

auc_score <- list()
auc_score[[1]] <- models[[1]]$score(msr("classif.auc"))
auc_score[[2]] <- models[[2]]$score(msr("classif.auc"))
auc_score[[3]] <- models[[3]]$score(msr("classif.auc"))
auc_score[[4]] <- models[[4]]$score(msr("classif.auc"))
auc_score[[5]] <- models[[5]]$score(msr("classif.auc"))
auc_score[[6]] <- models[[6]]$score(msr("classif.auc"))
