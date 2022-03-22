bpl=bank_personal_loan
skimr::skim(bpl)
DataExplorer::plot_bar(bpl, ncol = 3)
DataExplorer::plot_histogram(bpl, ncol = 3)
DataExplorer::plot_boxplot(bpl, by = "Personal.Loan", ncol = 3)

bpl=subset(bpl, select = -ZIP.Code)

library("data.table")
library("mlr3")
#####################log-reg
credit_fit <- glm(Personal.Loan ~ ., binomial, bpl)
#summary(credit_fit)
credit_pred <- predict(credit_fit, bpl)
credit_pred <- predict(credit_fit, bpl, type = "response")
y_hat <- factor(ifelse(credit_pred> 0.5, 1,0))
#Acc
mean(I(y_hat == bpl$Personal.Loan))
#Matrix
table(truth = bpl$Personal.Loan, prediction =y_hat)

################################
credit_lda <- MASS::lda(Personal.Loan ~ ., bpl)
credit_pred <- predict(credit_lda, na.omit(bpl))
mean(I(credit_pred$class == na.omit(bpl)$Personal.Loan))
table(truth = bpl$Personal.Loan, prediction =credit_pred$class) 


########################
library("mlr3learners")
library("mlr3proba")
bpl[, "Personal.Loan"] <- factor(bpl[, "Personal.Loan"])
set.seed(212) # set seed for reproducibility
credit_task <- TaskClassif$new(id = "BankCredit",backend = bpl,target = "Personal.Loan",positive = "0") 
cv5 <- rsmp("cv", folds = 5)
cv5$instantiate(credit_task)


#####################################
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Now try with a model that needs no missingness
lrn_log_reg <- lrn("classif.log_reg", predict_type = "prob")
pl_log_reg <- pl_missing %>>%
  po(lrn_log_reg)

res <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_baseline,
                    lrn_cart,
                    #lrn_cart_cp,
                    pl_xgb,
                    pl_log_reg),
  resampling = list(cv5)
), store_models = TRUE)

res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))

##########################
lrn_cart <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_0.01 <- lrn("classif.rpart", predict_type = "prob",cp=0.01)
lrn_cart_0.011 <- lrn("classif.rpart", predict_type = "prob",cp=0.011)
lrn_cart_0.012 <- lrn("classif.rpart", predict_type = "prob",cp=0.012)
res_baseline <- resample(credit_task, lrn_baseline, cv5, store_models = TRUE)
res_cart <- resample(credit_task, lrn_cart, cv5, store_models = TRUE)
res <- benchmark(data.table(
  task       = list(credit_task),
  learner    = list(lrn_cart,
                    lrn_cart_0.01,
                    lrn_cart_0.011,
                    lrn_cart_0.012
  ),
  
  resampling = list(cv5)
), store_models = TRUE)
res$aggregate(list(msr("classif.ce"),
                   msr("classif.acc"),
                   msr("classif.auc"),
                   msr("classif.fpr"),
                   msr("classif.fnr")))
trees <- res$resample_result(2)
# Then, let's look at the tree from first CV iteration, for example:
tree1 <- trees$learners[[1]]
# This is a fitted rpart object, so we can look at the model within
tree1_rpart <- tree1$model
# If you look in the rpart package documentation, it tells us how to plot the
# tree that was fitted
plot(tree1_rpart, compress = TRUE, margin = 0.1)
text(tree1_rpart, use.n = TRUE, cex = 0.4)
####################################
lrn_cart_cv <- lrn("classif.rpart", predict_type = "prob", xval = 10)
res_cart_cv <- resample(credit_task, lrn_cart_cv, cv5, store_models = TRUE)
rpart::plotcp(res_cart_cv$learners[[5]]$model)

#######################################3
# Define a collection of base learners
lrn_baseline <- lrn("classif.featureless", predict_type = "prob")
lrn_cart     <- lrn("classif.rpart", predict_type = "prob")
lrn_cart_cp  <- lrn("classif.rpart", predict_type = "prob", cp = 0.016, id = "cartcp")
lrn_ranger   <- lrn("classif.ranger", predict_type = "prob")
lrn_xgboost  <- lrn("classif.xgboost", predict_type = "prob")
lrn_log_reg  <- lrn("classif.log_reg", predict_type = "prob")

# Define a super learner
lrnsp_log_reg <- lrn("classif.log_reg", predict_type = "prob", id = "super")

# Missingness imputation pipeline
pl_missing <- po("fixfactors") %>>%
  po("removeconstants") %>>%
  po("imputesample", affect_columns = selector_type(c("ordered", "factor"))) %>>%
  po("imputemean")

# Factors coding pipeline
pl_factor <- po("encode")

# Now define the full pipeline
spr_lrn <- gunion(list(
  # First group of learners requiring no modification to input
  gunion(list(
    po("learner_cv", lrn_baseline),
    po("learner_cv", lrn_cart),
    po("learner_cv", lrn_cart_cp)
  )),
  # Next group of learners requiring special treatment of missingness
  pl_missing %>>%
    gunion(list(
      po("learner_cv", lrn_ranger),
      po("learner_cv", lrn_log_reg),
      po("nop") # This passes through the original features adjusted for
      # missingness to the super learner
    )),
  # Last group needing factor encoding
  pl_factor %>>%
    po("learner_cv", lrn_xgboost)
)) %>>%
  po("featureunion") %>>%
  po(lrnsp_log_reg)

# This plot shows a graph of the learning pipeline
spr_lrn$plot()
#################################
res_spr <- resample(credit_task, spr_lrn, cv5, store_models = TRUE)
res_spr$aggregate(list(msr("classif.ce"),
                       msr("classif.acc"),
                       msr("classif.fpr"),
                       msr("classif.fnr")))
