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

dl_dummy_task <- dataToTask("credit_card_prediction/dummy_data/dl_dummy_data.csv", "dl_dummy")
dl_oh_task <- dataToTask("credit_card_prediction/oh_data/dl_oh_data.csv", "dl_oh")
dl_iv_task <- dataToTask("credit_card_prediction/iv_data/dl_iv_data.csv", "dl_iv")

mf_dummy_task <- dataToTask("credit_card_prediction/dummy_data/mf_dummy_data.csv", "mf_dummy")
mf_oh_task <- dataToTask("credit_card_prediction/oh_data/mf_oh_data.csv", "mf_oh")
mf_iv_task <- dataToTask("credit_card_prediction/iv_data/mf_iv_data.csv", "mf_iv")

mice_dummy_task <- dataToTask("credit_card_prediction/dummy_data/mice_dummy_data.csv", "mice_dummy")
mice_oh_task <- dataToTask("credit_card_prediction/oh_data/mice_oh_data.csv", "mice_oh")
mice_iv_task <- dataToTask("credit_card_prediction/iv_data/mice_iv_data.csv", "mice_iv")


dl <- list(dummy=dl_dummy_task, oh=dl_oh_task, iv=dl_iv_task)
mf <- list(dummy=mf_dummy_task, oh=mf_oh_task, iv=mf_iv_task)
mice <- list(dummy=mice_dummy_task, oh=mice_oh_task, iv=mice_iv_task)

tasks <- list(dl=dl, mf=mf, mice=mice)

rm(dl, mf, mice)
rm(dl_dummy_task, mf_dummy_task, mice_dummy_task)
rm(dl_oh_task, mf_oh_task, mice_oh_task)
rm(dl_iv_task, mf_iv_task, mice_iv_task)

# tasks[["dl"]][["dummy"]]
# tasks[["<type>"]][["<code>"]]