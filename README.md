<!DOCTYPE html>
<html>
    <h1>Time Series Analysis</h1>
    
<h3>Python ARIMA Model and Evaluation<h3>
_________________________________________________________________________________________________________________________________________________________________
>> <h4>AutoRegressive Integrated Moving Average (ARIMA) Model</h4>

### Introduction to ARIMA Models
So what exactly is an ARIMA model?

**ARIMA**, short for ‘Auto Regressive Integrated Moving Average’ is actually a class of models that <em>'explains’</em> a given time series based on its own past values, that is, its own lags and the lagged forecast errors (in Machine Learning terms), so that equation can be used to forecast future values.

Any <em>non-seasonal</em> time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.

An <strong>ARIMA model</strong> is characterized by 3 terms: *p*, *d*, and *q*,

where,
<ol>

> <li> <em>p</em> is the order of the <em>AR</em> term</li>
> <li> <em>q</em> is the order of the <em>MA</em> term</li>
> <li> <em>d</em> is the number of differencing required to make the time series stationary</li>
</ol>

So, what does the <strong><em>order of AR term</em></strong> even mean? Before we go there, let’s first look at the ‘d’ term.


<h3>What <em>p, d and q</em> in ARIMA model mean</h3>

The first step to build an ARIMA model is to make the time series stationary.

__Why?__

Because the term **‘Auto Regressive(AR)’** in ***AR__IMA*** means it is a linear regression model that uses its own lags as predictors. Linear regression models work best when the predictors are not correlated and are independent of each other.

**So how do we make a series stationary?**
**Differencing**
The most common approach is to difference it. That is, subtract the previous value from the current value. Sometimes, depending on the complexity of the series, more than one differencing may be needed.

The value of <em>d</em>, therefore, is the minimum number of differencing needed to make the series stationary. And if the time series is already stationary, then *d = 0*.

Next, what are the *‘p’* and *‘q’* terms?

**‘p’** is the order of the **‘Auto Regressive’ (AR)** term. It refers to the number of lags of *Y* to be used as predictors. And *‘q’* is the order of the **‘Moving Average’ (MA)** term. It refers to the number of lagged forecast errors that should go into the ARIMA Model.


A pure **Auto Regressive** (AR only) model is one where *Y_t* depends only on its own lags. That is, *Y_t* is a function of the ‘lags of *Y_t*’. 

Likewise a pure __Moving Average__ (MA only) model is one where *Y_t* depends only on the lagged forecast errors.

- where the error terms are the errors of the autoregressive models of the respective lags. 

An ARIMA model is one where the time series was differenced at least once to make it stationary and you combine the AR and the MA terms.
ARIMA model in words:

Predicted *Y_t* = Constant + Linear combination Lags of *Y* (upto *p* lags) + Linear Combination of Lagged forecast errors (upto *q* lags)

The objective, therefore, is to identify the values of *p*, *d* and *q*. But the real question is, how?

Let’s start with finding the *d*.

### How to find the order of differencing (*d*) in ARIMA model.

The purpose of differencing it to make the time series stationary.

But you need to be careful to not over-difference the series. Because, an over differenced series may still be stationary, which in turn will affect the model parameters.

So how to determine the right order of differencing?

The right order of differencing is the minimum differencing required to get a near-stationary series which roams around a defined mean and the ACF plot reaches to zero fairly quick.

If the autocorrelations are positive for many number of lags (10 or more), then the series needs further differencing. On the other hand, if the lag 1 autocorrelation itself is too negative, then the series is probably over-differenced.

In the event, you can’t really decide between two orders of differencing, then go with the order that gives the least standard deviation in the differenced series.

Let’s see how to do it with an example.

First, I am going to check if the series is stationary using the __Augmented Dickey Fuller test__ (`adfuller()`), from the statsmodels package.

Why?

Because, you need differencing only if the series is non-stationary. Else, no differencing is needed, that is, `d=0`.

The null hypothesis of the ADF test is that the time series is non-stationary. So, if the p-value of the test is less than the significance level (0.05) then you reject the null hypothesis and infer that the time series is indeed stationary.

So, in our case, if `p_value` > 0.05 we go ahead with finding the order of differencing.


---
`About the Dataset`
---
The dataset used in this notebook was supplied by The Lesotho Electricity Company, the sole distributor of electricity in Lesotho. 


---
Problem Statement
---
_'Aggregate Electricity Consumption/Sale'_ dataset, as mentioned above, comes from the Lesotho Electricity Company (LEC). This is a univariate timeseries dataset that describes the electricity monthly sales/consumption provided in Gigawatt hours from January 2004 to December 2018. The dataset contains only 2 columns, one column is Date and the other relates to Electricity Sales.

---

`Goal`
---
-  The goal is to predict electricity demand/consumption from 2019 till 2050.

---
</html> 


**Here are the steps to follow when building a Machine Learning Model**
--

- Data Exploration
- Data Preprocessing
- Splitting Data For Training and Testing
- Preparing ARIMA model
- Assembling all of the steps using pipeline
- Training the model
- Running predictions on the model
- Evaluating and visualizing model performance

