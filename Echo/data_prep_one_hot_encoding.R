## Data preparation using one hot encoder
install.packages("dataPreparation")
library(dataPreparation)

# Import data
setwd("/Users/echo/Desktop/i2ml_final_project")
source("/Users/echo/Desktop/i2ml_final_project/credit_card_prediction/data_preparation.R")
data_onehot <- final_data

# Impose one hot encoding on the Multicategorical variables such as GENDER, etc.

# one_hot_encoder(
#   dataSet, 
#   encoding,
#   verbose = TRUE,
#   drop = FALSE
# )
# Explanation
# dataSet: refers to the data frame which you wanna deal with
# encoding: Result of funcion build_encoding, (list, default to NULL).
#           To perform the same encoding on train and test, it is recommended to compute build_encoding before. If it is kept to NULL, build_encoding will be called.
# verbose: should the function log (logical, default to TRUE)
# drop: should cols be dropped after generation (logical, default to FALSE)

# Compute encoding
encoding <- build_encoding(data_onehot, cols = c("CODE_GENDER", "FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE"), verbose = TRUE)

# Apply one hot encoding
data_onehot <- one_hot_encoder(data_onehot, encoding = encoding, drop = TRUE)

# To see the output of categorical variables after one-hot encoding
# View(data_onehot)
# Result: turn the records of categories into 0,1 composed vectors
# convert all character y data into facotr datatype.
# data_onehot <- data %>%
#   mutate_if(is.character, as.factor) %>%
#   mutate(y = as.factor(y))
