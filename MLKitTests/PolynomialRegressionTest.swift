//
//  PolynomialRegressionTest.swift
//  MLKit
//
//  Created by Guled on 7/22/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import XCTest
import UIKit
@testable import MLKit
@testable import SwiftCSV
@testable import Upsurge

class PolynomialRegressionTests: XCTestCase {

    // (Might take awhile...) Tests fitting a model using gradient descent on 2 features from kc.csv (CSV containing sample dataset)
    func testFitGradientDescent() {

        // Obtain data from csv file
        let path = NSBundle(forClass: PolynomialRegressionTests.self).pathForResource("kc", ofType: "csv")
        let csvUrl = NSURL(fileURLWithPath: path!)
        let data = try! CSV(url: csvUrl)

        // Setup the features we need and convert them to floats if necessary
        let training_data_string = data.columns["sqft_living"]!
        let training_data_2_string = data.columns["bedrooms"]!
        // Features
        let training_data = training_data_string.map { Float($0)! }
        let training_data_2 = training_data_2_string.map { Float($0)! }

        // Output
        let output_as_string = data.columns["price"]!
        let output_data = output_as_string.map { Float($0)! }

        // Fit the model
        let polynomialModel = PolynomialLinearRegression()

        // Setup initial weights
        let initial_weights = Matrix<Float>(rows: 3, columns: 1, elements: [-100000.0, 1.0, 1.0])

        // Fit the model and obtain the weights
        let weights = try! polynomialModel.train([training_data, training_data_2], output: output_data, initial_weights: initial_weights, step_size: Float(4e-12), tolerance: Float(1e9))

        let actualWeights = Matrix<Float>(rows: 3, columns: 1, elements: [-99999.9, 303.321, 1.42297])

        XCTAssertEqualWithAccuracy(weights.column(0)[0], actualWeights.column(0)[0], accuracy: 0.1)
        XCTAssertEqualWithAccuracy(weights.column(0)[1], actualWeights.column(0)[1], accuracy: 0.01)
        XCTAssertEqualWithAccuracy(weights.column(0)[2], actualWeights.column(0)[2], accuracy: 0.01)
    }

    // (Might take awhile...) Tests RSS (cost function) for Polynomial Regression
    func testRSS() {
        // Obtain data from csv file
        let path = NSBundle(forClass: PolynomialRegressionTests.self).pathForResource("kc", ofType: "csv")
        let csvUrl = NSURL(fileURLWithPath: path!)
        let data = try! CSV(url: csvUrl)

        // Setup the features we need and convert them to floats if necessary
        let training_data_string = data.columns["sqft_living"]!
        let training_data_2_string = data.columns["bedrooms"]!
        // Features
        let training_data = training_data_string.map { Float($0)! }
        let training_data_2 = training_data_2_string.map { Float($0)! }

        // Output
        let output_as_string = data.columns["price"]!
        let output_data = output_as_string.map { Float($0)! }

        // Fit the model
        let polynomialModel = PolynomialLinearRegression()

        // Setup initial weights
        let initial_weights = Matrix<Float>(rows: 3, columns: 1, elements: [-100000.0, 1.0, 1.0])

        // Fit the model and obtain the weights
        let weights = try! polynomialModel.train([training_data, training_data_2], output: output_data, initial_weights: initial_weights, step_size: Float(4e-12), tolerance: Float(1e9))

        // Compute RSS
        let rss = try! polynomialModel.RSS([training_data, training_data_2], observation: output_data)

        // Estimated RSS
        let actualRSS = Float(1.48850266e+15)

        XCTAssertEqualWithAccuracy(rss, actualRSS, accuracy: 0.1)
    }

    // (Might take awhile...) Tests one-time prediction
    func testOneTimePrediction() {

        // Obtain data from csv file
        let path = NSBundle(forClass: PolynomialRegressionTests.self).pathForResource("kc", ofType: "csv")
        let csvUrl = NSURL(fileURLWithPath: path!)
        let data = try! CSV(url: csvUrl)

        // Setup the features we need and convert them to floats if necessary
        let training_data_string = data.columns["sqft_living"]!
        let training_data_2_string = data.columns["bedrooms"]!
        // Features
        let training_data = training_data_string.map { Float($0)! }
        let training_data_2 = training_data_2_string.map { Float($0)! }

        // Output
        let output_as_string = data.columns["price"]!
        let output_data = output_as_string.map { Float($0)! }

        // Fit the model
        let polynomialModel = PolynomialLinearRegression()

        // Setup initial weights
        let initial_weights = Matrix<Float>(rows: 3, columns: 1, elements: [-100000.0, 1.0, 1.0])

        // Fit the model and obtain the weights
        let weights = try! polynomialModel.train([training_data, training_data_2], output: output_data, initial_weights: initial_weights, step_size: Float(4e-12), tolerance: Float(1e9))

        // Make a prediction
        let quickPrediction = polynomialModel.predict([Float(1.0), Float(1180.0), Float(1.0)], your_weights: weights.elements)

        let estimatedPrediction = Float(257920.625)

        XCTAssertEqualWithAccuracy(quickPrediction, estimatedPrediction, accuracy: 0.0001)
    }

}