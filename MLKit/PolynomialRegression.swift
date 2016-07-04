//
//  PolynomialRegression.swift
//  MLKit
//
//  Created by Guled on 7/2/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//


// I NEED TO MAKE 
// RSS 
// POLYNOMIAL DEGREE FEATURE WHEN I LEARN ABOUT THE DISCRETE FEATURE STUFF [ DONE THROUGH MLDATAMANAGER


import Foundation
import Upsurge

public class PolynomialLinearRegression {
    
    public var cost_function_result:Float!
    public var weights:Matrix<Float>!
    
    public init(){
        cost_function_result = 0.0
    }
    
    
    
    public func fit(features:[Array<Float>], output: Array<Float>, initial_weights:Matrix<Float>, step_size:Float, tolerance:Float) -> Matrix<Float> {
        
        var converged = false
        var predictions:ValueArray<Float>!
        var errors:ValueArray<Float> = ValueArray<Float>()
        var weights = initial_weights
        var gradient_sum_of_squares = Float(0.0)
        var derivative = Float(0.0)
        
        
        // Convert the users array of features and output into matrices and vectors 
        let feature_matrix_and_output = MLDataManager.dataToMatrix(features, output: output)
        let feature_matrix = feature_matrix_and_output.0
        let output_vector = feature_matrix_and_output.1
        
        // Get Feature Derivative
        let get_feature_derivative = { (errors:ValueArray<Float>, feature:ValueArraySlice<Float>) -> Float in
            let derivative = 2 * (errors • feature)
            
            return derivative
        }
        
        while !converged {
            
            predictions = predictEntireMatrixOfFeatures(feature_matrix, weights: weights)
            errors = output_vector - predictions
            
            gradient_sum_of_squares = Float(0.0)
            
            for i in 0...weights.count-1 {
            
                derivative = get_feature_derivative(errors, feature_matrix.column(i))
                gradient_sum_of_squares = gradient_sum_of_squares + (derivative * derivative)
                
               weights.elements[i] = weights.elements[i] + (step_size * derivative)
                
            }
            
            gradient_sum_of_squares = sqrt(gradient_sum_of_squares)
            
            if gradient_sum_of_squares < tolerance {
                converged = true
            }
        }
        
        // Set weights 
        self.weights = weights
        
        return weights
    }
    
    
    
    public func RSS(features:[Array<Float>], observation: Array<Float>) -> Float {
        // Check if the users model has fit to their data
        if self.weights == nil {
            print("You need to have fit a model first before computing the RSS/Cost Function.")
            return Float(-1)
        }
        
        
        // First get the predictions
        let y_actual = observation
        let feature_matrix_and_output = MLDataManager.dataToMatrix(features, output: observation)
        let feature_matrix = feature_matrix_and_output.0
        let y_predicted = predictEntireMatrixOfFeatures(feature_matrix, weights: self.weights)
        
        // Then compute the residuals/errors
        let error:ValueArray<Float> = (y_actual - y_predicted)
        
        // Then square and add them up
        var result = dot(error, error)
        
        // Set cost function
        self.cost_function_result = result
        
        return result
    }

    public func predict(input_vector:ValueArray<Float>, weights:ValueArray<Float>) -> Float {
        
        let prediction:Float = input_vector • weights
        
        return prediction
    }
    
    public func predictEntireMatrixOfFeatures(input_matrix:Matrix<Float>, weights:Matrix<Float>) -> ValueArray<Float> {
        
        let predictions = input_matrix * weights

        return predictions.elements
    }
    
}