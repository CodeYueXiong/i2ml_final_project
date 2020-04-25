library(mlr3)
library(mlr3learners)

# ----------------------------------------
# ------- train one task with one learner
# ----------------------------------------

train_model <- function(task, learner, resampling){
  set.seed(2020)
  print(task$id)
  train_model <- resample(task, learner, resampling, store_models = TRUE)
  print(train_model)
}

# ----------------------------------------
# - select feautures
# ----------------------------------------

select_tasks_features <- function(tasks, features){
  new_tasks <- tasks
  miss_name <- c('dl', 'mf', 'mice')
  code_name <- c('dummy', 'oh', 'iv')
  for(missing in miss_name){
    for(coding in code_name){
      new_tasks[[missing]][[coding]] = tasks[[missing]][[coding]]$select(features)
    }
  }
  select_tasks_features <- new_tasks
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


evaluate_models <- function(models){
  par(mfrow=c(3,3))
  for(m in models){
    name <- m$task$id
    auc <- m$aggregate(msr("classif.auc"))[[1]]
    max_auc <- max(m$score(msr("classif.auc"))[,9])
    print(sprintf("%10s: %.4f (max: %.4f)", name, auc, max_auc))
    #cat(paste0(name, ": ", auc, "\t(max: ", max_auc, ")\n"))
  }
}

multiplot_roc <- function(models){
  plots <- list()
  k <- 1
  for(m in models){
    name <- m$task$id
    plots[[k]] <- autoplot(m, type = "roc") + xlab("") + ylab("") + ggtitle(name)
    k <- k+1
  }
  do.call("grid.arrange", c(plots))  
}


