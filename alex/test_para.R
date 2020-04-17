# set.seed(2020) before training
# the bigger k is, the larger the difference in AUC
# distance increase runtime, but not significent better
# scale=TRUR, longer runtime, but not significent better (also make difference more noticeable)

# mf best in general
# encoding no big difference

#learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=5, distance=2, scale=FALSE)
# > evalutate_models(models)
# [1] "  dl_dummy: 0.7447 (max: 0.7921)"
# [1] "     dl_oh: 0.7447 (max: 0.7921)"
# [1] "     dl_iv: 0.7447 (max: 0.7921)"
# [1] "  mf_dummy: 0.7443 (max: 0.7612)"
# [1] "     mf_oh: 0.7443 (max: 0.7611)"
# [1] "     mf_iv: 0.7443 (max: 0.7612)"
# [1] "mice_dummy: 0.7443 (max: 0.7611)"
# [1] "   mice_oh: 0.7443 (max: 0.7611)"
# [1] "   mice_iv: 0.7443 (max: 0.7612)"



# ***********************************
# **************************** best?
# ***********************************

# learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=2, scale=FALSE)
# > evalutate_models(models)
# [1] "  dl_dummy: 0.7445 (max: 0.7839)"
# [1] "     dl_oh: 0.7445 (max: 0.7839)"
# [1] "     dl_iv: 0.7445 (max: 0.7841)"
# [1] "  mf_dummy: 0.7520 (max: 0.7884)"
# [1] "     mf_oh: 0.7520 (max: 0.7884)"
# [1] "     mf_iv: 0.7520 (max: 0.7885)"
# [1] "mice_dummy: 0.7519 (max: 0.7884)"
# [1] "   mice_oh: 0.7519 (max: 0.7884)"
# [1] "   mice_iv: 0.7520 (max: 0.7886)"


# learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=20, distance=2, scale=FALSE)
# > evalutate_models(models)
# [1] "  dl_dummy: 0.7435 (max: 0.7913)"
# [1] "     dl_oh: 0.7435 (max: 0.7914)"
# [1] "     dl_iv: 0.7434 (max: 0.7913)"
# [1] "  mf_dummy: 0.7494 (max: 0.7890)"
# [1] "     mf_oh: 0.7494 (max: 0.7889)"
# [1] "     mf_iv: 0.7494 (max: 0.7890)"
# [1] "mice_dummy: 0.7494 (max: 0.7890)"
# [1] "   mice_oh: 0.7494 (max: 0.7889)"
# [1] "   mice_iv: 0.7494 (max: 0.7890)"

#----------- scale=TRUE

#learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=2, scale=TRUE)
# > evalutate_models(models)
# [1] "  dl_dummy: 0.7331 (max: 0.7711)"
# [1] "     dl_oh: 0.7388 (max: 0.7787)"
# [1] "     dl_iv: 0.7188 (max: 0.7726)"
# [1] "  mf_dummy: 0.7052 (max: 0.7367)"
# [1] "     mf_oh: 0.7078 (max: 0.7351)"
# [1] "     mf_iv: 0.7012 (max: 0.7424)"
# [1] "mice_dummy: 0.7069 (max: 0.7531)"
# [1] "   mice_oh: 0.7066 (max: 0.7424)"
# [1] "   mice_iv: 0.6971 (max: 0.7318)"

#----------- distance=3

#learner <- lrn("classif.kknn", id = "knn", predict_type = "prob", k=15, distance=3, scale=FALSE)
# > evalutate_models(models)
# [1] "  dl_dummy: 0.7446 (max: 0.7842)"
# [1] "     dl_oh: 0.7446 (max: 0.7842)"
# [1] "     dl_iv: 0.7446 (max: 0.7842)"
# [1] "  mf_dummy: 0.7512 (max: 0.7884)"
# [1] "     mf_oh: 0.7512 (max: 0.7884)"
# [1] "     mf_iv: 0.7512 (max: 0.7885)"
# [1] "mice_dummy: 0.7512 (max: 0.7884)"
# [1] "   mice_oh: 0.7512 (max: 0.7884)"
# [1] "   mice_iv: 0.7513 (max: 0.7886)"

# ------------------------------------------------
# ---------------------------- feature selection
# ------------------------------------------------

