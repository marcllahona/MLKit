//
//  RidgeRegression.swift
//  MLKit
//
//  Created by Guled on 7/10/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation
import Upsurge


public class RidgeRegression {
    
    public var cost_function_result:Float!
    public var weights: Matrix<Float>!
    
    public init() {
        cost_function_result = 0.0
    }
    
    
    /**
     The fit method fits your model and returns your regression coefficients/weights. The methods applies gradient descent
     as a means to find the most optimal regression coefficients for your model.
     
     - parameter features: An array of all of your features.
     - parameter output: An array of your observations/output.
     - parameter intial_weights: A row or column matrix of your initial weights. If you have no initial weights simply pass in a zero matrix of type Matrix (check the Upsurge framework for details on this type).
     - parameter step_size: The amount of "steps" you want to take in the gradient descent process.
     - parameter max_iterations: Defines the maximum number of iterations and takes gradient steps (based on your step_size) until we reach this maximum number.
     
     - returns: A Matrix of type Float consisting your regression coefficients/weights.
     */
    public func fit(features: [Array<Float>], output: Array<Float>, initial_weights: Matrix<Float>, step_size: Float, l2_penalty:Float, max_iterations:Float = 100) throws -> Matrix<Float> {
        
        // Error Handeling
        
        // Check Feature Length
        var featureLength = 0
        
        for (i, featureArray) in features.enumerate() {
            
            if i == 0 {
                featureLength = featureArray.count
            }
            
            if featureArray.count != featureLength {
                throw RegressionError.lengthOfDataArrayNotEqual
            }
        }
        
        
        // Main ML Algorithm
        var predictions: ValueArray<Float>!
        var errors: ValueArray<Float> = ValueArray<Float>()
        let weights = initial_weights
        var derivative = Float(0.0)
        var iterations = Float(0.0)
        
        
        // Convert the users array of features and output into matrices and vectors
        let feature_matrix_and_output = MLDataManager.dataToMatrix(features, output: output)
        let feature_matrix = feature_matrix_and_output.0
        let output_vector = feature_matrix_and_output.1
        
        //while not reached maximum number of iterations:
        while iterations < max_iterations {
            
            // compute the predictions based on feature_matrix and weights
            predictions = predictEntireMatrixOfFeatures(feature_matrix, weights: weights)
            
            // compute the errors as predictions - output
            errors = predictions - output_vector

            for i in 0...weights.count - 1 {
                
                // compute the derivative for weight[i].
                if i == 0 {
                    derivative = getFeatureDerivative(errors, feature: feature_matrix.column(i), weight: weights.elements[i], l2_penalty: l2_penalty, feature_is_constant: false)
                }else{
                    derivative = getFeatureDerivative(errors, feature: feature_matrix.column(i), weight: weights.elements[i], l2_penalty: l2_penalty, feature_is_constant: true)
                }
                
                // subtract the step size times the derivative from the current weight
                weights.elements[i] = weights.elements[i] - (step_size * derivative)
            }
            
            iterations = iterations + 1
        }
        
        // Set weights
        self.weights = weights
        
        return weights
    }
    
    
    /**
     The RSS method computes the residual sum of squares or the cost function of your model.
     
     - parameter features: An array of numbers.
     - parameter observation: An array of your observations/output.
     - returns: The cost of your model (a.k.a The Residual Sum of Squares).
     */
    public func RSS(features: [Array<Float>], observation: Array<Float>) throws -> Float {
        // Check if the users model has fit to their data
        if self.weights == nil {
            print("You need to have fit a model first before computing the RSS/Cost Function.")
            throw RegressionError.modelHasNotBeenFit
        }
        
        // First get the predictions
        let y_actual = observation
        let feature_matrix_and_output = MLDataManager.dataToMatrix(features, output: observation)
        let feature_matrix = feature_matrix_and_output.0
        let y_predicted = predictEntireMatrixOfFeatures(feature_matrix, weights: self.weights)
        
        // Then compute the residuals/errors
        let error: ValueArray<Float> = (y_actual - y_predicted)
        
        // Then square and add them up
        let result = dot(error, error)
        
        // Set cost function
        self.cost_function_result = result
        
        return result
    }

    /**
     The predict method is used for making one-time predictions by passing in an input vector and the weights you have generated
     when fitting your model (using the fit() method).
     
     - parameter input_vector: An array of input (depends on how many features you used to fit your model)
     - parameter weights: An array of your weights. This can be obtained by fitting your model before getting a prediction.
     - returns: A prediction (of type Float).
     */
    public func predict(input_vector: ValueArray<Float>, weights: ValueArray<Float>) -> Float {
        
        let prediction: Float = input_vector • weights
        
        return prediction
    }
    
    /**
     The predictEntireMatrixOfFeatures is used for making predictions using all of your feature data.
     
     - parameter input_matrix: You need to utilize the dataToMatrix() method from the MLDataManager class in order to convert your array of features into a matrix. input_matrix is a
     matrix that consists of all of your features.
     - parameter weights: A matrix that consists of your weights.
     - returns: An array of predictions (of type Float)
     */
    public func predictEntireMatrixOfFeatures(input_matrix: Matrix<Float>, weights: Matrix<Float>) -> ValueArray<Float> {
        
        let predictions = input_matrix * weights
        
        return predictions.elements
    }
    
    
    func getFeatureDerivative(errors: ValueArray<Float>, feature:ValueArraySlice<Float>, weight:Float, l2_penalty: Float, feature_is_constant:Bool) -> Float{
        
     
        var derivative = Float(0)
        
        if feature_is_constant {
            let sum_of_predictions_and_output = (2 * (errors • feature) )
            let product_of_l2penalty_and_weights = 2 * (l2_penalty * weight)
            derivative =  sum_of_predictions_and_output  +  product_of_l2penalty_and_weights
        }else{
            let sum_of_predictions_and_output = (2 * (errors • feature) )
            derivative = sum_of_predictions_and_output
        }
        
        return derivative
    }
    
    
    // K-Fold Cross Validation Coming Soon!
    
    
}