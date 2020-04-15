# import data

setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")
source("credit_card_prediction/data_preparation.R")
data <- final_data

# check type and distribution of all columns
str(data)
summary(data)

# --------------------------------------------
# ----------------------- dummy variable start
# --------------------------------------------

# A dummy variable is a numeric variable that represents nominal variable by only taking the value 0 or 1,
# indicating presence or absence of the category.

# import library to transform factor varible into dummy variable
library(fastDummies)

# Since the library 'fastDummies' tansforms all factor variable into dummy variables,
# we will convert our target "y" (factor) into a character variable
# to avoid it being transformed to dummy variable.
data$y <- as.numeric(as.character(data$y)) 

# transform all factor variables to dummy variables, and removes the original variables that were used to generate the dummy variables.
data_dummy <- fastDummies::dummy_cols(data, remove_selected_columns=TRUE) 

# column name convention fix (mlr3 name convention - space to underscore)
library(janitor)
data_dummy <- clean_names(data_dummy)

# remove dummy variable created by NA occupation ("occupation_type_na")
data_dummy <- within(data_dummy, rm(occupation_type_na))
data_dummy <- as.data.frame(sapply(data_dummy, as.numeric))

str(data_dummy)

# --------------------------------------------
# ----------------------- dummy variable end
# --------------------------------------------


# --------------------------------------------
# ----------------------- handle missing data
# ----------------------- OCCUPATION_TYPE
# --------------------------------------------

# remove na
dl_dummy_data <- na.omit(data_dummy)

# miss forest
library(missForest)
mf_dummy_data <- missForest(data_dummy, maxiter=1)[[1]]

# sum all OCCUPATION_TYPE column
count <- rowSums(select(mf_dummy_data, c(40:57)))
sum(count > 1) # 10845
sum(count == 1) # 25134
sum(count < 1) # 478
sum(count == 0) # 0
sum(count != 0) # 36457

# MICE
# Error in colMeans(as.matrix(imp[[j]]), na.rm = TRUE) : 
#  'x' must be numeric

# defaultMethod = c("pmm", "logreg", "polyreg", "polr")
# defaultMethod A vector of length 4 containing the default imputation methods for 
# 1) numeric data, 
# 2) factor data with 2 levels, 
# 3) factor data with > 2 unordered levels, and 
# 4) factor data with > 2 ordered levels. 
# By default, the method uses pmm, 
# predictive mean matching (numeric data) logreg, 
# logistic regression imputation (binary data, factor with 2 levels) polyreg, 
# polytomous regression imputation for unordered categorical data (factor > 2 levels) polr, proportional odds model for (ordered, > 2 levels).

library(mice)
mice_dummy_data <- data_dummy
mice_dummy_data <- mice(mice_dummy_data, m=1, method="pmm", seed = 2020, maxiter=1)
mice_dummy_data <- mice::complete(mice_dummy_data, 1)


# --------------- TODO: do mice before dummy with "polyreg"
mice_raw_data <- data
mice_raw_data <- mice(mice_raw_data, m=1, method="polyreg", seed = 2020)
mice_raw_data <- mice::complete(mice_raw_data, 1)
write_csv2(mice_raw_data,"credit_card_prediction/dummy_data/mice_raw_data.csv")


#data_dummy$y <- as.factor(data_dummy$y)
write_csv2(dl_dummy_data,"credit_card_prediction/dummy_data/dl_dummy_data.csv")
write_csv2(mf_dummy_data,"credit_card_prediction/dummy_data/mf_dummy_data.csv")
write_csv2(mice_dummy_data,"credit_card_prediction/dummy_data/mice_dummy_data.csv")



