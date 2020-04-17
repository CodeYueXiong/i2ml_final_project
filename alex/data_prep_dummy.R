library(fastDummies)
library(janitor)


setwd("C:/Users/user/Documents/R-projects/i2ml_final_project")

# A dummy variable is a numeric variable that represents nominal variable by only taking the value 0 or 1,
# indicating presence or absence of the category.

dummy_var <- function(data){
  
  # Since the library 'fastDummies' tansforms all factor variable into dummy variables,
  # we will convert our target "y" (factor) into a character variable
  # to avoid it being transformed to dummy variable.
  
  data$y <- as.numeric(as.character(data$y)) 
  
  # transform all factor variables to dummy variables, and removes the original variables that were used to generate the dummy variables.
  data_dummy <- fastDummies::dummy_cols(data, remove_selected_columns=TRUE) 
  
  # column name convention fix (mlr3 name convention - space to underscore)
  data_dummy <- clean_names(data_dummy)
  
  
  data_dummy <- as.data.frame(sapply(data_dummy, as.numeric))
  data_dummy$y <- as.factor(data_dummy$y)
  
  dummy_var <- data_dummy
}


# --------------------------------------------
# ----------------------- handle missing data
# ----------------------- OCCUPATION_TYPE
# --------------------------------------------


dl_dummy_data <- read.csv2("credit_card_prediction/dl_na_data.csv", header = TRUE)
dl_dummy_data <- dummy_var(dl_dummy_data)

mf_dummy_data <- read.csv2("credit_card_prediction/mf_na_data.csv", header = TRUE)
mf_dummy_data <- dummy_var(mf_dummy_data)

mice_dummy_data <- read.csv2("credit_card_prediction/mice_na_data.csv", header = TRUE)
mice_dummy_data <- dummy_var(mice_dummy_data)


# --------------------------------------------
# --------------------- output data into file
# --------------------------------------------

write_csv2(dl_dummy_data,"credit_card_prediction/dummy_data/dl_dummy_data.csv")
write_csv2(mf_dummy_data,"credit_card_prediction/dummy_data/mf_dummy_data.csv")
write_csv2(mice_dummy_data,"credit_card_prediction/dummy_data/mice_dummy_data.csv")


