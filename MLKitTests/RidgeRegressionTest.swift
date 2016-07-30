//
//  RidgeRegressionTest.swift
//  MLKit
//
//  Created by Guled on 7/29/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import XCTest
import UIKit
@testable import MLKit
@testable import SwiftCSV
@testable import Upsurge

class RidgeRegressionTests: XCTestCase {

    func testFitGradientDescent() {
        // Obtain data from csv file
        let path = NSBundle(forClass: PolynomialRegressionTests.self).pathForResource("kc", ofType: "csv")
        let csvUrl = NSURL(fileURLWithPath: path!)
        let data = try! CSV(url: csvUrl)

        // Setup the features we need and convert them to floats if necessary
        let training_data_string = data.columns["sqft_living"]!
        // Features
        let training_data = training_data_string.map { Float($0)! }

        // Output
        let output_as_string = data.columns["price"]!
        let output_data = output_as_string.map { Float($0)! }

        // Fit the model
        let ridgeModel = RidgeRegression()

        // Setup initial weights
        let initial_weights = Matrix<Float>(rows: 2, columns: 1, elements: [0.0, 0.0])

        // Fit the model and obtain the weights
        let weights = try! ridgeModel.fit([training_data], output: output_data, initial_weights: initial_weights, step_size: Float(1e-12), l2_penalty: 0.0, max_iterations: 1000)

        let actualWeights: ValueArray<Float> = [-0.201522, 263.089]

        XCTAssertEqualWithAccuracy(weights.elements[0], actualWeights[0], accuracy: 0.01)
        XCTAssertEqualWithAccuracy(weights.elements[1], actualWeights[1], accuracy: 0.01)
    }

    func testRSS() {
        // Obtain data from csv file
        let path = NSBundle(forClass: PolynomialRegressionTests.self).pathForResource("kc", ofType: "csv")
        let csvUrl = NSURL(fileURLWithPath: path!)
        let data = try! CSV(url: csvUrl)

        // Setup the features we need and convert them to floats if necessary
        let training_data_string = data.columns["sqft_living"]!
        // Features
        let training_data = training_data_string.map { Float($0)! }

        // Output
        let output_as_string = data.columns["price"]!
        let output_data = output_as_string.map { Float($0)! }

        // Fit the model
        let ridgeModel = RidgeRegression()

        // Setup initial weights
        let initial_weights = Matrix<Float>(rows: 2, columns: 1, elements: [0.0, 0.0])

        // Fit the model and obtain the weights
        _ = try! ridgeModel.fit([training_data], output: output_data, initial_weights: initial_weights, step_size: Float(1e-12), l2_penalty: 0.0, max_iterations: 1000)

        // Calculate RSS
        let RSS = try! ridgeModel.RSS([training_data], observation: output_data)
        print(RSS)
        let actualRSS: Float = 1.48397428e+15
        XCTAssertEqual(RSS, actualRSS)

    }

    func testOneTimePrediction() {
        //1.18000000e+03
        
        // Obtain data from csv file
        let path = NSBundle(forClass: PolynomialRegressionTests.self).pathForResource("kc", ofType: "csv")
        let csvUrl = NSURL(fileURLWithPath: path!)
        let data = try! CSV(url: csvUrl)
        
        // Setup the features we need and convert them to floats if necessary
        let training_data_string = data.columns["sqft_living"]!
        // Features
        let training_data = training_data_string.map { Float($0)! }
        
        // Output
        let output_as_string = data.columns["price"]!
        let output_data = output_as_string.map { Float($0)! }
        
        // Fit the model
        let ridgeModel = RidgeRegression()
        
        // Setup initial weights
        let initial_weights = Matrix<Float>(rows: 2, columns: 1, elements: [0.0, 0.0])
        
        // Fit the model and obtain the weights
        let weights = try! ridgeModel.fit([training_data], output: output_data, initial_weights: initial_weights, step_size: Float(1e-12), l2_penalty: 0.0, max_iterations: 1000)
        
        // Make a prediction
        let quickPrediction = ridgeModel.predict([Float(1.0), Float(1.18000000e+03)], your_weights: weights.elements)
        print(quickPrediction)
        let actualPrediction:Float = 310445.062
        
        XCTAssertEqual(quickPrediction, actualPrediction)

    }

}