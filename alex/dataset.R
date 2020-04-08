# import data
cred_rd <- read.csv(file = 'credit_card_prediction/credit_record.csv')
app_rd <- read.csv(file = 'credit_card_prediction/application_record.csv')

# check matches
sum(app_rd$ID %in% cred_rd$ID, na.rm = TRUE) # 36457
sum(cred_rd$ID %in% app_rd$ID, na.rm = TRUE) # 777715

# merge data by ID (inner join)
data <- merge(cred_rd, app_rd, by="ID")

# check NA
summary(cred_rd)
summary(app_rd)
summary(data)
