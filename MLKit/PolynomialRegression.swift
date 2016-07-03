//
//  PolynomialRegression.swift
//  MLKit
//
//  Created by Guled on 7/2/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation
import Upsurge

public class PolynomialLinearRegression {
    
    public var cost_function_result:Float!
    
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
        
        
        return weights
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