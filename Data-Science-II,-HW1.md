Data Science II, HW1
================

``` r
  library(glmnet)
  library(caret)
  library(dplyr)
  library(tidymodels) # mata-engines for model training
  library(corrplot)   # to generate correlation matrix
  library(ggplot2)    # for plots and graphics
  library(plotmo)     # to generate trace plot 
  library(ggrepel)    # for plotting functions
  library(pls)
```

``` r
  train <- read.csv("../housing_training.csv") %>% na.omit() %>% relocate("Sale_Price","Overall_Qual","Kitchen_Qual","Fireplace_Qu","Exter_Qual")
  test <- read.csv("../housing_test.csv") %>% na.omit() %>% relocate("Sale_Price","Overall_Qual","Kitchen_Qual","Fireplace_Qu","Exter_Qual")
```

``` r
  set.seed(21)
```

``` r
  predictors = model.matrix(Sale_Price ~ ., train)[,-1]  # matrix of predictors, converted categorical to dummies
  response   = train[, "Sale_Price"]
  corrplot::corrplot(cor(predictors), method = "shade", type = "upper", tl.cex=.75, tl.col="black") # heatmap
```

<img src="Data-Science-II,-HW1_files/figure-gfm/Matrix of Predictors-1.png" style="display: block; margin: auto;" />

``` r
  lambda_grid <- exp(seq(10,-5, length = 100))  # lambda sequence in DESC order
```

### Part A — Lasso Model

``` r
  cv.lasso = cv.glmnet(x=predictors, y=response, standardize=TRUE, alpha=1, lambda=lambda_grid)
  plot(cv.lasso)
```

<img src="Data-Science-II,-HW1_files/figure-gfm/Lasso Regression-1.png" style="display: block; margin: auto;" />

``` r
  cv.lasso$lambda.min
```

    ## [1] 37.95357

``` r
  cv.lasso$lambda.1se
```

    ## [1] 580.3529

``` r
  plot_glmnet(cv.lasso$glmnet.fit)
```

<img src="Data-Science-II,-HW1_files/figure-gfm/Lasso Regression-2.png" style="display: block; margin: auto;" />

``` r
  predict(cv.lasso, s="lambda.min", type="coefficients")
```

    ## 40 x 1 sparse Matrix of class "dgCMatrix"
    ##                               lambda.min
    ## (Intercept)                -4.907777e+06
    ## Overall_QualAverage        -4.940427e+03
    ## Overall_QualBelow_Average  -1.262353e+04
    ## Overall_QualExcellent       7.442639e+04
    ## Overall_QualFair           -1.096786e+04
    ## Overall_QualGood            1.220173e+04
    ## Overall_QualVery_Excellent  1.337492e+05
    ## Overall_QualVery_Good       3.796778e+04
    ## Kitchen_QualFair           -2.571422e+04
    ## Kitchen_QualGood           -1.796666e+04
    ## Kitchen_QualTypical        -2.603274e+04
    ## Fireplace_QuFair           -7.731067e+03
    ## Fireplace_QuGood            .           
    ## Fireplace_QuNo_Fireplace    1.997372e+03
    ## Fireplace_QuPoor           -5.705537e+03
    ## Fireplace_QuTypical        -7.004799e+03
    ## Exter_QualFair             -3.509091e+04
    ## Exter_QualGood             -1.671808e+04
    ## Exter_QualTypical          -2.114334e+04
    ## Gr_Liv_Area                 6.570689e+01
    ## First_Flr_SF                7.892018e-01
    ## Second_Flr_SF               .           
    ## Total_Bsmt_SF               3.532596e+01
    ## Low_Qual_Fin_SF            -4.132301e+01
    ## Wood_Deck_SF                1.178914e+01
    ## Open_Porch_SF               1.570662e+01
    ## Bsmt_Unf_SF                -2.089667e+01
    ## Mas_Vnr_Area                1.071695e+01
    ## Garage_Cars                 4.131482e+03
    ## Garage_Area                 8.026780e+00
    ## Year_Built                  3.235195e+02
    ## TotRms_AbvGrd              -3.698223e+03
    ## Full_Bath                  -4.036119e+03
    ## Fireplaces                  1.089817e+04
    ## Lot_Frontage                1.007723e+02
    ## Lot_Area                    6.047300e-01
    ## Longitude                  -3.381597e+04
    ## Latitude                    5.670486e+04
    ## Misc_Val                    8.661587e-01
    ## Year_Sold                  -5.945218e+02

- **Selected Tuning Parameter for Lasso** (optimal λ): 38

  - **Corresponding Test RMSE**: 6.1606

- **\#of Predictors associates with 1SE**: 36

### Part B — Elastic Net

``` r
  TC = trainControl(method="cv", number=10)
  ENet.fit <- train( Sale_Price ~ ., 
                     data=train,
                     method="glmnet", 
                     standardize = TRUE,
                     tuneGrid= expand.grid(alpha=round(seq(0,1,length=20), 2),
                                           lambda=lambda_grid),
                     trControl = TC
                    )
  
  myPar = list(superpose.symbol = list(col=rainbow(25)),
               superpose.line   = list(col=rainbow(25)))
  plot(ENet.fit, par.settings=myPar, xTrans=log)
```

<img src="Data-Science-II,-HW1_files/figure-gfm/Elastic Net Model-1.png" style="display: block; margin: auto;" />

``` r
  ENet_RMSE <- sqrt(mean((test$Sale_Price - predict(ENet.fit, newdata = test))^2))
```

``` r
  min_rmse <- min(ENet.fit$results$RMSE) # Find the minimum RMSE
  rmse_1se_threshold <- min_rmse + sd(ENet.fit$results$RMSE) # Compute 1SE threshold
  lambda_1se <- max(ENet.fit$results$lambda[ENet.fit$results$RMSE <= rmse_1se_threshold])
  subset_1se <- ENet.fit$results[ENet.fit$results$lambda == lambda_1se, ] # Subset rows where lambda=lambda_1se
  alpha_1se <- subset_1se$alpha[which.min(subset_1se$RMSE)] # Select the alpha that gives the lowest RMSE within that subset
  
  ENet_1SE <- glmnet(as.matrix(train[, -which(names(train) == "Sale_Price")]), 
                              train$Sale_Price, 
                              alpha = alpha_1se, 
                              lambda = lambda_1se, 
                              standardize = TRUE)
```

    ## Warning in storage.mode(xd) <- "double": NAs introduced by coercion

``` r
  y_pred_elastic_1se <- predict(ENet_1SE, newx = as.matrix(test[, -which(names(test) == "Sale_Price")]))
```

    ## Warning in cbind2(1, newx) %*% nbeta: NAs introduced by coercion

``` r
  rmse_elastic_1se <- sqrt(mean((test$Sale_Price - y_pred_elastic_1se)^2))
```

- **Optimal:**
  - α = 0.05
  - λ = 580.3529
  - **Corresponding Test Error RMSE:** 20939.7
- **Yes, 1SE rule is applicable.**
  - α =0
  - λ =22026.47
  - **Corresponding Test Error RMSE:** 28953.52

### Part C

``` r
  PLS.fit <- train(
    Sale_Price ~ .,
    data = train,
    method = "pls",
    trControl = trainControl(method = "cv", number = 10),
    tuneLength = 20  # Allows automatic selection of the optimal number of components
  )
```

``` r
  best_ncomp <- PLS.fit$bestTune$ncomp
  cat("Optimal number of PLS components:", best_ncomp, "\n")
```

    ## Optimal number of PLS components: 20

``` r
  pls_final <- plsr(Sale_Price ~ ., data = train, ncomp = best_ncomp, validation = "CV")
```

``` r
  y_pred_pls <- predict(pls_final, newdata = test, ncomp = best_ncomp)
  rmse_pls <- sqrt(mean((test$Sale_Price - y_pred_pls)^2))
```

- **Optimal \#of PLS Components =** 20

- **Test RMSE =** 23406.07

### Part D

``` r
  Lasso.fit <- train(Sale_Price ~ ., 
                   data=train,
                   method="glmnet",
                   standardize = TRUE,
                   tuneGrid= expand.grid(alpha=1, lambda=lambda_grid),
                   trControl = TC)
  
  lm.fit = train(Sale_Price~., 
                 data=train,
                 method="lm",
                 trControl=TC)
  
  resamp = resamples(list( enet=ENet.fit, lasso=Lasso.fit, lm=lm.fit ))
  summary(resamp)
```

    ## 
    ## Call:
    ## summary.resamples(object = resamp)
    ## 
    ## Models: enet, lasso, lm 
    ## Number of resamples: 10 
    ## 
    ## MAE 
    ##           Min.  1st Qu.   Median     Mean  3rd Qu.     Max. NA's
    ## enet  14947.11 15876.67 16657.02 16675.32 17318.31 18840.61    0
    ## lasso 14723.80 15958.44 16669.72 16619.33 17277.95 18303.18    0
    ## lm    14718.54 15573.92 16464.55 16739.60 17644.55 19593.76    0
    ## 
    ## RMSE 
    ##           Min.  1st Qu.   Median     Mean  3rd Qu.     Max. NA's
    ## enet  19879.73 20618.00 22661.16 23026.20 24835.31 29124.12    0
    ## lasso 19211.95 21595.46 22705.51 22771.61 23446.15 26510.12    0
    ## lm    19567.23 21518.59 23125.73 22982.74 24597.28 25661.64    0
    ## 
    ## Rsquared 
    ##            Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
    ## enet  0.8821751 0.8994086 0.9029030 0.9043826 0.9134120 0.9187349    0
    ## lasso 0.8744909 0.8853843 0.9028549 0.9036043 0.9194009 0.9385559    0
    ## lm    0.8778431 0.8903336 0.9050049 0.9024935 0.9143747 0.9227287    0

- **Comparing the RMSE among Models we get:**

  - Elastic Net: 22,998.34

  - Lasso: 22,791.35

  - LM: 22,858.80

    **Lasso has the lowest RMSE, making it the best predictor based on
    this metric.**

- **Comparing R2:**

  - Elastic Net: 0.9003

  - Lasso: 0.9009

  - LM: 0.9059

    **Linear regression has the highest R², meaning it explains more
    variance, but its RMSE is slightly higher than Lasso. Also, this
    difference is very small among three models.**

- **Final Conclusion: Lasso Model provides the best trade-off between
  error minimization and model fit.**

### Part E

``` r
  Lasso.fit <- train(Sale_Price ~ ., 
                   data=train,
                   method="glmnet",
                   standardize = TRUE,
                   tuneGrid= expand.grid(alpha=1, lambda=lambda_grid),
                   trControl = TC)

  min_rmse <- min(Lasso.fit$results$RMSE)

  lambda_min <- Lasso.fit$bestTune$lambda
  
# Compute the 1SE threshold
  rmse_1se_threshold <- min_rmse + sd(Lasso.fit$results$RMSE)

# Select the largest lambda within 1SE
  lambda_1se <- max(Lasso.fit$results$lambda[Lasso.fit$results$RMSE <= rmse_1se_threshold])
```

- **Optimal:**
  - λ = 32.6175
- **1SE:**
  - λ =4160.262
- **Discussion on the Discrepancies:**
  - Both methods are valid, just using different techniques.
  - Unlike caret::train(), cv.glmnet() has its own built-in selection
    process, affecting in how it searches for the best and 1se λ.
  - The first method caret::train() tends to select a larger λ1SE,
    potentially leading to a simpler model. The second methodcv.glmnet()
    may provide a finer control over λ selection, making it more stable.
  - In cv.glmnet(), lambda.1SE is calculated within cv.glmnet()’s
    internal cross-validation process. **In** caret::train(), we
    manually compute lambda.1se, and if the lambda grid is different,
    the highest λ within 1SE can be significantly larger.
