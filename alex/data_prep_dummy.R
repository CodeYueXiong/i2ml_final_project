# import data
setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
source("credit_card_prediction/data_preparation.R")
data <- final_data

# check type of all colums
str(data)

# data prep: dummy variable
library(fastDummies)
data_dummy <- fastDummies::dummy_cols(data, remove_selected_columns=TRUE) # removes the columns used to generate the dummy columns 
str(data_dummy)

# simple model
model <- glm(y~.-ID, data=data_dummy)
summary(model)
