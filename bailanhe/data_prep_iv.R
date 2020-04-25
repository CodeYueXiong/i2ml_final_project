library(tidyverse)
library(data.table)
library(checkmate)
library(mltools)
library(missForest)
library(mice)
library(scorecard)

set.seed(2020)
setwd("./i2ml programm/final project/i2ml_final_project/credit_card_prediction")


## calc_iv function compute the "information value" of a variable.
calc_iv <- function(feature) {
  # how many rows we need.
  number_row <- length(unique(final_data[[feature]]))
  # initialize the tibble
  inner_iv_table <- tibble(feature = 0, val = 0, all = 0, good = 0, bad = 0)
  # for each level, we calculate their information value
  for (i in seq(number_row)) {
    # mark which level is computed
    val <- unique(final_data[[feature]])[[i]]
    # conver data as "data.table" for the convenience of select.
    final_data <- data.table(final_data)
    # compute how many simples within this level
    all <- nrow(final_data[final_data[[feature]] == val])
    # the number of good people within this level
    good <- nrow(final_data[final_data[[feature]] == val & final_data$y == 0])
    # the number of bad people within this level
    bad <- nrow(final_data[final_data[[feature]] == val & final_data$y == 1])
    # store as a tibble
    inner_tibble <- tibble(feature, val, all, good, bad)
    # rbind "inner_tibble" with "inner_iv_table
    inner_iv_table <- rbind(inner_iv_table, inner_tibble)
  }
  # delet our initial rows.
  inner_iv_table <- inner_iv_table[-1, ]
  
  # compute IV
  inner_iv_table <- inner_iv_table %>%
    # compute the proportion for each level,
    mutate(share = all / sum(all)) %>%
    # compute bad rate
    mutate(bad_rate = bad / all) %>%
    # compute good distribution
    mutate(good_dis = (all - bad) / (sum(all) - sum(bad))) %>%
    # compute bad distribution
    mutate(bad_dis = bad / sum(bad)) %>%
    # compute woe
    mutate(woe = log(good_dis / bad_dis))
  # deal with the extreme situation
  inner_iv_table$woe[is.infinite(inner_iv_table$woe)] <- 0
  # compute IV
  iv_table <- inner_iv_table %>% mutate(iv = woe * (good_dis - bad_dis))
  # return value of IV
  print(sum(iv_table$iv))
  # return "iv_table"
  return(iv_table)
}


convert_binary_variable <- function(feature) {
  # save condition for succinct expression
  # check for defensive programm
  if (check_class(feature, "factor") & nlevels(feature) == 2) {
    levels(feature) <- seq(nlevels(feature))
    feature <- as.numeric(feature)
    return(feature)
  }
}
#whole process of dl_iv_data
#########
##change final_data
dl_data <- read.csv2("./dl_na_data.csv")
final_data <- dl_data
# convert all binary variable into binary number
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)

# find which variables have several values
dif_variable <- setdiff(names(final_data), names(converted_data))

####################### analyse the details of first multi-factors variable
calc_iv(dif_variable[1])
# found the variable "NAME_INCOME_TYPE" has 5 types of values, we integrate
# "student" , "pensioner" with "state servant"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "State servant", "Pensioner" = "State servant")

# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

# integrate "student" , "pensioner" with "state servant" seems not good.
# try integrate "student" "pensioner" with "Commercial associate"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "Commercial associate", "Pensioner" = "Commercial associate")


# Iv increases from 0.01 to 0.013, seems better.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

####################### analyse the details of second multi-factors variable
calc_iv(dif_variable[2])
# found the variable "NAME_EDUCSTION_TYPE" has 5 types of values. we integrate
# "Academic degree" with "Higher education"
final_data["less_factor_edu"] <- final_data %>%
  pull(dif_variable[[2]]) %>%
  recode("Academic degree" = "Higher education")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[2])
calc_iv("less_factor_edu")

####################### analyse the details of third multi-factors variable
calc_iv(dif_variable[3])
final_data["less_factor_status"] <- final_data %>% pull(dif_variable[3])
# found this variable seems relatively balanced, no need to change.


####################### analyse the details of fourth multi-factors variable
calc_iv(dif_variable[4])
# found the variable "NAME_HOUSING_TYPE" has 6 types of values. we integrate
# "Co-op apartment" with "Office apartment"
final_data["less_factor_house"] <- final_data %>%
  pull(dif_variable[[4]]) %>%
  recode("Co-op apartment" = "Office apartment")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[4])
calc_iv("less_factor_house")

####################### analyse the details of fifth multi-factors variable
calc_iv(dif_variable[5])
# found the variable "NAME_HOUSING_TYPE" has too manny types of values. we integrate

final_data["less_factor_work"] <- final_data %>%
  pull(dif_variable[[5]]) %>%
  recode("Cleaning staff" = "Labor",
         "Cooking staff" = "Labor",
         "Drivers" = "Labor",
         "Laborers" = "Labor",
         "Low-skill Laborers" = "Labor",
         "Security staff" = "Labor",
         "Waiters/barmen staff" = "Labor") %>% 
  recode("Accountants" = "Office",
         "Core staff" = "Office",
         "HR staff" = "Office",
         "Medicine staff" = "Office",
         "Private service staff" = "Office",
         "Realty agents" = "Office",
         "Sales staff" = "Office",
         "Secretaries" = "Office") %>% 
  recode("Managers" = "higher",
         "High skill tech staff" = "higher",
         "IT staff" = "higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[5])
calc_iv("less_factor_work")

########
#encode multi-factors variables into one-hot 
factor_data <- final_data %>% select(starts_with("less")) %>% as.data.table()
oh_data <- one_hot(factor_data,dropCols = T)
#cbind oh_data with converted_data
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)
converted_data <- converted_data %>% add_column(oh_data)


#Binning continuous variable to improve performance.
dl_bin <- woebin(converted_data,y= "y",x=c("CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","DAYS_EMPLOYED","CNT_FAM_MEMBERS"), method = "chimerge")
#plot binning variable
plotlist = woebin_plot(dl_bin)
#cbind dl_bin with dl_iv_data
dl_iv_data = woebin_ply(converted_data, rbindlist(dl_bin))
#save dl_iv_data
dl_iv_data$y <- as.factor(dl_iv_data$y)
colnames(dl_iv_data) <- make.names(colnames(dl_iv_data),unique=T)
write_csv2(dl_iv_data,"../dl_iv_data.csv")

#########
#whole process of mf_iv_data
##########

#save mf_iv_data
mf_data <- read.csv2("./mf_na_data.csv")
final_data <- mf_data


# convert all binary variable into binary number
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)

# find which variables have several values
dif_variable <- setdiff(names(final_data), names(converted_data))

####################### analyse the details of first multi-factors variable
calc_iv(dif_variable[1])
# found the variable "NAME_INCOME_TYPE" has 5 types of values, we integrate
# "student" , "pensioner" with "state servant"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "State servant", "Pensioner" = "State servant")

# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

# integrate "student" , "pensioner" with "state servant" seems not good.
# try integrate "student" "pensioner" with "Commercial associate"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "Commercial associate", "Pensioner" = "Commercial associate")


# Iv increases from 0.01 to 0.013, seems better.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

####################### analyse the details of second multi-factors variable
calc_iv(dif_variable[2])
# found the variable "NAME_EDUCSTION_TYPE" has 5 types of values. we integrate
# "Academic degree" with "Higher education"
final_data["less_factor_edu"] <- final_data %>%
  pull(dif_variable[[2]]) %>%
  recode("Academic degree" = "Higher education")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[2])
calc_iv("less_factor_edu")

####################### analyse the details of third multi-factors variable
calc_iv(dif_variable[3])
final_data["less_factor_status"] <- final_data %>% pull(dif_variable[3])
# found this variable seems relatively balanced, no need to change.


####################### analyse the details of fourth multi-factors variable
calc_iv(dif_variable[4])
# found the variable "NAME_HOUSING_TYPE" has 6 types of values. we integrate
# "Co-op apartment" with "Office apartment"
final_data["less_factor_house"] <- final_data %>%
  pull(dif_variable[[4]]) %>%
  recode("Co-op apartment" = "Office apartment")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[4])
calc_iv("less_factor_house")

####################### analyse the details of fifth multi-factors variable
calc_iv(dif_variable[5])
# found the variable "NAME_HOUSING_TYPE" has too manny types of values. we integrate

final_data["less_factor_work"] <- final_data %>%
  pull(dif_variable[[5]]) %>%
  recode("Cleaning staff" = "Labor",
         "Cooking staff" = "Labor",
         "Drivers" = "Labor",
         "Laborers" = "Labor",
         "Low-skill Laborers" = "Labor",
         "Security staff" = "Labor",
         "Waiters/barmen staff" = "Labor") %>% 
  recode("Accountants" = "Office",
         "Core staff" = "Office",
         "HR staff" = "Office",
         "Medicine staff" = "Office",
         "Private service staff" = "Office",
         "Realty agents" = "Office",
         "Sales staff" = "Office",
         "Secretaries" = "Office") %>% 
  recode("Managers" = "higher",
         "High skill tech staff" = "higher",
         "IT staff" = "higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[5])
calc_iv("less_factor_work")
#########
#encode multi-factors variables into one-hot 
factor_data <- final_data %>% select(starts_with("less")) %>% as.data.table()
oh_data <- one_hot(factor_data,dropCols = T)
#cbind oh_data with converted_data
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)
converted_data <- converted_data %>% add_column(oh_data)
#Binning continuous variable to improve performance.
dl_bin <- woebin(converted_data,y= "y",x=c("CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","DAYS_EMPLOYED","CNT_FAM_MEMBERS"), method = "chimerge")
#plot binning variable
plotlist = woebin_plot(dl_bin)
#cbind dl_bin with dl_iv_data
mf_iv_data = woebin_ply(converted_data, rbindlist(dl_bin))
#save mf_iv_data
mf_iv_data$y <- as.factor(mf_iv_data$y)
colnames(mf_iv_data) <- make.names(colnames(mf_iv_data),unique=T)
write_csv2(mf_iv_data,"../mf_iv_data.csv")
#########
#whole process of mice_iv_data
#########
#change final_data
mice_data <- read.csv2("./mice_na_data.csv")
final_data <- mice_data
# convert all binary variable into binary number
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)

# find which variables have several values
dif_variable <- setdiff(names(final_data), names(converted_data))

####################### analyse the details of first multi-factors variable
calc_iv(dif_variable[1])
# found the variable "NAME_INCOME_TYPE" has 5 types of values, we integrate
# "student" , "pensioner" with "state servant"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "State servant", "Pensioner" = "State servant")

# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

# integrate "student" , "pensioner" with "state servant" seems not good.
# try integrate "student" "pensioner" with "Commercial associate"
final_data["less_factor_income"] <- final_data %>%
  pull(dif_variable[[1]]) %>%
  recode("Student" = "Commercial associate", "Pensioner" = "Commercial associate")


# Iv increases from 0.01 to 0.013, seems better.
calc_iv(dif_variable[1])
calc_iv("less_factor_income")

####################### analyse the details of second multi-factors variable
calc_iv(dif_variable[2])
# found the variable "NAME_EDUCSTION_TYPE" has 5 types of values. we integrate
# "Academic degree" with "Higher education"
final_data["less_factor_edu"] <- final_data %>%
  pull(dif_variable[[2]]) %>%
  recode("Academic degree" = "Higher education")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[2])
calc_iv("less_factor_edu")

####################### analyse the details of third multi-factors variable
calc_iv(dif_variable[3])
final_data["less_factor_status"] <- final_data %>% pull(dif_variable[3])
# found this variable seems relatively balanced, no need to change.


####################### analyse the details of fourth multi-factors variable
calc_iv(dif_variable[4])
# found the variable "NAME_HOUSING_TYPE" has 6 types of values. we integrate
# "Co-op apartment" with "Office apartment"
final_data["less_factor_house"] <- final_data %>%
  pull(dif_variable[[4]]) %>%
  recode("Co-op apartment" = "Office apartment")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[4])
calc_iv("less_factor_house")

####################### analyse the details of fifth multi-factors variable
calc_iv(dif_variable[5])
# found the variable "NAME_HOUSING_TYPE" has too manny types of values. we integrate

final_data["less_factor_work"] <- final_data %>%
  pull(dif_variable[[5]]) %>%
  recode("Cleaning staff" = "Labor",
         "Cooking staff" = "Labor",
         "Drivers" = "Labor",
         "Laborers" = "Labor",
         "Low-skill Laborers" = "Labor",
         "Security staff" = "Labor",
         "Waiters/barmen staff" = "Labor") %>% 
  recode("Accountants" = "Office",
         "Core staff" = "Office",
         "HR staff" = "Office",
         "Medicine staff" = "Office",
         "Private service staff" = "Office",
         "Realty agents" = "Office",
         "Sales staff" = "Office",
         "Secretaries" = "Office") %>% 
  recode("Managers" = "higher",
         "High skill tech staff" = "higher",
         "IT staff" = "higher")
# analyse how the IV of the variable has been changed.
calc_iv(dif_variable[5])
calc_iv("less_factor_work")
#########
#encode multi-factors variables into one-hot 
factor_data <- final_data %>% select(starts_with("less")) %>% as.data.table()
oh_data <- one_hot(factor_data,dropCols = T)
#cbind oh_data with converted_data
converted_data <- final_data %>% mutate_if(is.factor, convert_binary_variable)
converted_data <- converted_data %>% add_column(oh_data)
#Binning continuous variable to improve performance.
dl_bin <- woebin(converted_data,y= "y",x=c("CNT_CHILDREN","AMT_INCOME_TOTAL","DAYS_BIRTH","DAYS_EMPLOYED","CNT_FAM_MEMBERS"), method = "chimerge")
#plot binning variable
plotlist = woebin_plot(dl_bin)
#cbind dl_bin with dl_iv_data
mice_iv_data = woebin_ply(converted_data, rbindlist(dl_bin))
#save mice_iv_data
mice_iv_data$y <- as.factor(mice_iv_data$y)
colnames(mice_iv_data) <- make.names(colnames(mice_iv_data),unique=T) 
write_csv2(mice_iv_data,"../mice_iv_data.csv")




