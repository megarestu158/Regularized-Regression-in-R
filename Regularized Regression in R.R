# Description: HW Day 21
# Author: Mega Restu Utari
# Date: January 2022

# import data
data <- read.csv("boston.csv")

# SPLIT DATA into 3 parts
# train - validation - test
library(caTools)
set.seed(123)
sample <- sample.split(data$medv, SplitRatio = .80)
pre_train <- subset(data, sample == TRUE)
sample_train <- sample.split(pre_train$medv, SplitRatio = .80)

# train-validation data
train <- subset(pre_train, sample_train == TRUE)
validation <- subset(pre_train, sample_train == FALSE)

# test data
test <- subset(data, sample == FALSE)

# DRAW CORRELATION
library(psych)
pairs.panels(train, 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE, # show correlation ellipses
             cex.cor = 3
) 
# Observation:
## RAD and TAX has the max correlation of 0.9, so we can remove either one of them.
## We choose TAX because it has greater correlation with the target feature than with RAD.

# drop correlated columns
library(dplyr)
drop_cols <- c('rad')

train <- train %>% select(-drop_cols)
validation <-  validation %>% select(-drop_cols)
test <- test %>% select(-drop_cols)

# FIT MODEL (ridge and LASSO)
# feature preprocessing
# to ensure we handle categorical features
x <- model.matrix(medv ~ ., train)[,-1]
y <-  train$medv

# ridge regression
# fit multiple ridge regression with different lambda
# lambda = [0.01, 0.1, 1, 10]
library(glmnet)
ridge_reg_pointzeroone <- glmnet(x, y, alpha = 0, lambda = 0.01)
coef(ridge_reg_pointzeroone)

ridge_reg_pointone <- glmnet(x, y, alpha = 0, lambda = 0.1)
coef(ridge_reg_pointone)

ridge_reg_one <- glmnet(x, y, alpha = 0, lambda = 1)
coef(ridge_reg_one)

ridge_reg_ten <- glmnet(x, y, alpha = 0, lambda = 10)
coef(ridge_reg_ten)

# lasso regression
# fit multiple lasso regression with different lambda
# lambda = [0.01, 0.1, 1, 10]
lasso_reg_pointzeroone <- glmnet(x, y, alpha = 1, lambda = 0.01)
coef(lasso_reg_pointzeroone) 

lasso_reg_pointone <- glmnet(x, y, alpha = 1, lambda = 0.1)
coef(lasso_reg_pointone) 

lasso_reg_one <- glmnet(x, y, alpha = 1, lambda = 1)
coef(lasso_reg_one)

lasso_reg_ten <- glmnet(x, y, alpha = 1, lambda = 10)
coef(lasso_reg_ten)


# CHOOSE THE BEST LAMBDA FROM VALIDATION SET

# Make predictions on the validation data
x_validation <- model.matrix(medv ~., validation)[,-1]
y_validation <- validation$medv

RMSE_ridge_pointzeroone <- sqrt(mean((y_validation - predict(ridge_reg_pointzeroone, x_validation))^2))
RMSE_ridge_pointzeroone # 4.3464 (best)

RMSE_ridge_pointone <- sqrt(mean((y_validation - predict(ridge_reg_pointone, x_validation))^2))
RMSE_ridge_pointone # 4.349494 

RMSE_ridge_one <- sqrt(mean((y_validation - predict(ridge_reg_one, x_validation))^2))
RMSE_ridge_one # 4.422032

RMSE_ridge_ten <- sqrt(mean((y_validation - predict(ridge_reg_ten, x_validation))^2))
RMSE_ridge_ten # 5.342122

# Best model's coefficients
# recall the best model --> ridge_reg_pointzeroone
coef(ridge_reg_pointzeroone)
## Sample coeff interpretation:
## An increase of 1 point in RM, while the other 
## features are kept fixed, is associated with an 
## increase of 4.517287 point in MEDV.

# comparison on validation data
# to choose the best lambda
# Make predictions on the validation data
RMSE_lasso_pointzeroone <- sqrt(mean((y_validation - predict(lasso_reg_pointzeroone, x_validation))^2))
RMSE_lasso_pointzeroone # 4.340783 (best)

RMSE_lasso_pointone <- sqrt(mean((y_validation - predict(lasso_reg_pointone, x_validation))^2))
RMSE_lasso_pointone # 4.352728 

RMSE_lasso_one <- sqrt(mean((y_validation - predict(lasso_reg_one, x_validation))^2))
RMSE_lasso_one # 4.937774

RMSE_lasso_ten <- sqrt(mean((y_validation - predict(lasso_reg_ten, x_validation))^2))
RMSE_lasso_ten # 9.371755

# Best model's coefficients
# recall the best model --> lasso_reg_pointzeroone
coef(lasso_reg_pointzeroone)
## Sample coeff interpretation:
## An increase of 1 point in RM, while the other 
## features are kept fixed, is associated with an 
## increase of 4.531757 point in MEDV.

## EVALUATING THE MODEL
# true evaluation on test data
x_test <- model.matrix(medv ~., test)[,-1]
y_test <- test$medv

# Ridge

# RMSE
RMSE_ridge_best <- sqrt(mean((y_test - predict(ridge_reg_pointzeroone, x_test))^2))
RMSE_ridge_best #6.820639
##Interpretation:
##The standard deviation of prediction errors is 6.820639, i.e. from the regression 
##line, the residuals mostly deviate between +- 6.820639

# MAE
MAE_ridge_best <- mean(abs(y_test-predict(ridge_reg_pointzeroone, x_test)))
MAE_ridge_best #3.896186
## Interpretation:
## On average, our prediction deviates the true MEDV by 3.896186

# MAPE
MAPE_ridge_best <- mean(abs((predict(ridge_reg_pointzeroone, x_test) - y_test))/y_test)
MAPE_ridge_best #0.1710101
##Interpretation::
## Moreover, this 3.896186 is equivalent to 17% deviation relative to the true MEDV

# Lasso

# RMSE
RMSE_lasso_best <- sqrt(mean((y_test - predict(lasso_reg_pointzeroone, x_test))^2))
RMSE_lasso_best #6.823445
##Interpretation:
##The standard deviation of prediction errors is 6.823445, i.e. from the regression 
##line, the residuals mostly deviate between +- 6.823445

# MAE
MAE_lasso_best <- mean(abs(y_test-predict(lasso_reg_pointzeroone, x_test)))
MAE_lasso_best #3.888415
## Interpretation:
## On average, our prediction deviates the true MEDV by 3.888415

# MAPE
MAPE_lasso_best <- mean(abs((predict(lasso_reg_pointzeroone, x_test) - y_test))/y_test)
MAPE_lasso_best #0.1707025
##Interpretation::
## Moreover, this 3.888415 is equivalent to 17% deviation relative to the true MEDV

