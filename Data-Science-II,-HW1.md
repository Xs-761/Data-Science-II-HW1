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

  - **Corresponding Test MSE**: 37.9536

- **\#of Predictors associates with 1SE**: 36

### Part B — Elastic Net

``` r
  TC = trainControl(method="cv", number=10)
  ENet.fit <- train( Sale_Price ~ ., 
                     data=train,
                     method="glmnet", 
                     tuneGrid= expand.grid(alpha=seq(0,1,length=20), lambda=lambda_grid),
                     trControl = TC
                    )
```

- **Optimal Tuning Parameter given by:**
  - α = 0.0526
  - λ = 580.3529

### Part C

### Part D

### Part E
