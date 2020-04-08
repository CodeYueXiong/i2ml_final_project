library(tidyverse)

setwd("/Users/hebailan/R Programm/i2ml programm/final project/i2ml_final_project/credit_card_prediction")

application_data <- read_csv("application_record.csv")
record <- read_csv("credit_record.csv")

# features engineering


# ascertain when all users open their account.
# ascertain how long their account has been opened
# and rename the variable as opentime
opentime_data <- record %>%
  group_by(ID) %>%
  filter(MONTHS_BALANCE == min(MONTHS_BALANCE)) %>%
  select(ID, MONTHS_BALANCE) %>%
  rename(opentime = MONTHS_BALANCE)
# merge two dataset.
data_with_day <- left_join(application_data, opentime_data, by = "ID")
# creat task by changing "STATUS" into binary variable
#choose users who overdue for more than 60 days as target risk users.
#marked as "T", otherwise "F"
creat_target <- function(x) {
  if (x == "2" | x == "3" | x == "4" | x == "5") {
    return(TRUE)
  } else {
    return(FALSE)
  }
}
#compute variable "target" with function creat_target,save it in new_record 
new_record <- record %>% mutate(target = map_dbl(record$STATUS, creat_target))
# sum the value of variable "target" for each group(grouped by ID).
data_target <- new_record %>%
  group_by(ID) %>%
  summarise(y = sum(target))
# for each ID, if the target value >0, means there is at least one TRUE under the ID number,
#if an ID has one TURE, means this person has been overdue at least 60 days.
#then we mark this ID as 1,means we will not approve the application.
data_target$y <- map_dbl(data_target$y, function(x) ifelse(x > 0, 1, 0))
#merge two data with methord inner_join.
data <- inner_join(data_with_day, data_target, by = "ID")
#convert all character data into facotr datatype.
final_data <- data %>% mutate_if(is.character, as.factor)
