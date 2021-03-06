# [WIP] MLKit (a.k.a Machine Learning Kit) 🤖
MLKit is a simple machine learning framework written in Swift. Currently MLKit features machine learning algorithms that deal with the topic of regression, but the framework will expand over time with topics such as classification, clustering, recommender systems, and deep learning. The vision and goal of this framework is to provide developers with a toolkit to create products that can learn from data. MLKit is a side project of mine in order to make it easier for developers to implement machine learning algorithms on the go, and to familiarlize myself with machine learning concepts. 

## Features (So Far)

- [x] Matrix and Vector Operations (uses [Upsurge framework](https://github.com/aleph7/Upsurge))
- [x] Simple Linear Regression (Allows for 1 feature set)
- [x] Polynomial Regression (Allows for multiple features)
- [x] Ridge Regression
- [x] Lasso Regression
- [x] Allows for splitting your data into training, validation, and test sets. 
- [x] K-Fold Cross Validation & Ability to test various L2 penalties for Ridge Regression
- [x] Unit tests for SimpleLinearRegression, PolynomialRegression, RidgeRegression and MLDatamanager classes 

## Up Next
- [ ] Logistic Regression
- [ ] Binary Decision Tree 
- [ ] Example Project 
- [ ] More detailed README and tutorials

## Code Snippets / Tutorials
For tutorials and code snippets please read the [wiki](https://github.com/Somnibyte/MLKit/wiki). 

# License
MIT License

Copyright (c) 2016 Guled Ahmed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
