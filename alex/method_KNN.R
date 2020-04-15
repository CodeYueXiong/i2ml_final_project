# KNN

# simple model
#model <- glm(y~.-ID, data=data_dummy)
#summary(model)

# install.packages("mlr3learners", dependencies = TRUE)
library(mlr3)
library("mlr3learners")

# create task, learn and cross validation
creditTask <- TaskClassif$new(id = "data_dummy", backend = data_dummy, target = "y")
learner <- lrn("classif.kknn", id = "dummy", predict_type = "prob")
resampling = rsmp("cv", folds = 3)

# train -> CV
rr = resample(creditTask, learner, resampling, store_models = TRUE)
print(rr)

# evaluate error rate
rr$score(msr("classif.ce"))
rr$aggregate(msr("classif.ce")) # average

# plot resampling result
library("mlr3viz")
library("precrec")

autoplot(rr)
autoplot(rr, type = "roc")

# AUC
rr$score(msr("classif.auc"))
rr$aggregate(msr("classif.auc"))
