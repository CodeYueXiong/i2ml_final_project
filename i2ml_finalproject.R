# library packages
library(tidyverse)
library(readxl)
library(rlang)
# get & set work dir
getwd()
setwd("../final project")
# read data
data <- read_excel("dat_final.xlsx")
# function that changes two variables to our binary target
# input: first_variable,second_variable: name of variables, dtype = str
make_target <- function(first_variable, second_variable) {
  #select two variables from our data
  select(data, first_variable, second_variable) %>%
    map(diff, lag = 1) %>% #caculate first order of diff.
    #if diff. < 0, means the numbers go down, mark as 1, otherwise as 0
    map(function(x) ifelse(x < 0, 1, 0)) %>% 
    # change as tibble(a type of data.frame) for the next function (transmute)
    as.tibble() %>%
    #compute new column "target" = first_variable + second_variable ,drop other variables
    transmute(target = .data[[first_variable]] + .data[[second_variable]]) %>%
    #if target == 2, means both variables go down, sign 1, otherwise 0.
    map(function(x) ifelse(x == 2, 1, 0)) %>%
    #change as tibble(a type of data.frame)
    as.tibble()
}

# test for function make_target
# a <- select(data, DGS1,SP500) %>%
#   map(diff, lag = 1) %>%
#   map(function(x) ifelse(x < 0, 1, 0)) %>%
#   as.tibble()
# b <- select(data, DGS1, SP500) %>%
#   map(diff, lag = 1) %>%
#   map(function(x) ifelse(x < 0, 1, 0)) %>%
#   as.tibble() %>%
#   transmute(target = DGS1 + SP500) %>%
#   map(function(x) ifelse(x == 2, 1, 0)) %>%
#   as.tibble()

#function to make the whole dataset.
make_data <- function(data, first_variable, second_variable){
  #select all the variables except variables used to compute target
  select(data,-first_variable,-second_variable) %>%
    #we will lose one row because we calculate diff.
    slice(-nrow(data)) %>%
    #add y to the remaining variable.
    add_column(y = make_target(first_variable,second_variable))
}

#get 3 diff. target
y_1 <- make_target("DGS1","NASDAQCOM")
y_2 <- make_target("DGS1", "SP500")
y_3 <- make_target("DGS1", "SPASTT01USM657N")

#make datasets contain: target "y" and remaing variables.
#warning: lose one row because we calculate diff.
data_DGS_NASD <- make_data(data,"DGS1","NASDAQCOM") 
data_DGS_SP <- make_data(data, "DGS1", "SP500")
data_DGS_SPA <- make_data(data, "DGS1", "SPASTT01USM657N")

