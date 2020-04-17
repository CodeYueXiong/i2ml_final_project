library(mlr3)
library("mlr3learners")

# ----------------------------------------
# ------- train one task with one learner
# ----------------------------------------

train_model <- function(task, learner, resampling){
  print(task$id)
  train_model <- resample(task, learner, resampling, store_models = TRUE)
  print(train_model)
}


# ----------------------------------------
# - train multiple tasks with one learner
# ----------------------------------------

train_all <- function(tasks, learner, resampling){
  models <- list()
  miss_name <- c('dl', 'mf', 'mice')
  code_name <- c('dummy', 'oh', 'iv')
  
  for(missing in miss_name){
    for(coding in code_name){
      name <- paste0(missing, "_", coding)
      task <- tasks[[missing]][[coding]]
      models[[name]] <- train_model(task, learner, resampling)
    }
  }
  train_all <- models
}

# ----------------------------------------
# -------------- visualize trianed model
# ----------------------------------------

evaluate_result <- function(model){
  # evaluate error rate
  print(model$score(msr("classif.ce")))
  print(model$aggregate(msr("classif.ce"))) # average
  
  # plot resampling result
  plot1 <- autoplot(model)
  plot2 <- autoplot(model, type = "roc")
  
  # AUC
  print(model$score(msr("classif.auc")))
  print(model$aggregate(msr("classif.auc")))
  
  evaluate_result <- list(plot(plot1), plot(plot2))
}