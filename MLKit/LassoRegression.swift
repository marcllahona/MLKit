//
//  LassoRegression.swift
//  MLKit
//
//  Created by Guled on 7/30/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation
import Upsurge

public class LassoRegression {

    private var cost_function_result: Float!
    private var final_weights: Matrix<Float>!

    public init () {
        cost_function_result = 0.0
    }
    
    
    
    public func train(features: [Array<Float>], output: Array<Float>, initial_weights: Matrix<Float>, l1_penalty:Float, tolerance:Float) throws -> Matrix<Float> {
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
        
        
        // Convert the users array of features and output into matrices and vectors
        let feature_matrix_and_output = MLDataManager.dataToMatrix(features, output: output)
        let feature_matrix = feature_matrix_and_output.0
        let normalized_feature_matrix = transpose(normalize(feature_matrix))
        
        var converged:Bool = false
        
        while converged == false {
            
            var change_for_full_cycle:[Float] = []
            
            for i  in (0..<initial_weights.elements.count) {
                
                let old_weight_i = initial_weights.elements[i]

                initial_weights.elements[i] = lassoCoordinateDescentStep(i, feature_matrix: normalized_feature_matrix, output: output, weights: initial_weights, l1_penalty: l1_penalty)
                
                let change = abs(initial_weights.elements[i] - old_weight_i)
                
                change_for_full_cycle.append(change)
            }
            
            let max_change = max(change_for_full_cycle)

            if max_change < tolerance {
                converged = true
            }
        }
        
        // set the weights
        self.final_weights = initial_weights

        return initial_weights
    }
    
    
    
    func lassoCoordinateDescentStep(i: Int, feature_matrix: Matrix<Float>, output: Array<Float>, weights: Matrix<Float>, l1_penalty: Float) -> Float {

        // Compute predictions
        let predictions = predictEntireMatrixOfFeatures(feature_matrix, your_weights: weights)

        // Compute ro[i]
        let ro_i_as_value_array: ValueArray<Float> = feature_matrix.column(i) * ((output - predictions) + weights.elements[i] * feature_matrix.column(i))

        let ro_i = sum(ro_i_as_value_array)

        // Calculate new weight
        var new_weight:Float! = 0.0

        if i == 0 {
            new_weight = ro_i
        }else if ro_i < (-l1_penalty/2.0) {
            new_weight = (ro_i + l1_penalty/2.0)
        }else if ro_i > (l1_penalty/2.0) {
            new_weight = (ro_i - l1_penalty/2.0)
        }else{
            new_weight = 0.0
        }
        
        return new_weight
    }
    

    /**
     The RSS method computes the residual sum of squares or the cost function of your model.

     - parameter features: An array of numbers. Your features will automatically be normalized.
     - parameter observation: An array of your observations/output.
     - returns: The cost of your model (a.k.a The Residual Sum of Squares).
     */
    public func RSS(features: [Array<Float>], observation: Array<Float>) throws -> Float {
        // Check if the users model has fit to their data
        if self.final_weights == nil {
            print("You need to have fit a model first before computing the RSS/Cost Function.")
            throw RegressionError.modelHasNotBeenFit
        }

        // First get the predictions
        let y_actual = observation
        let feature_matrix_and_output = MLDataManager.dataToMatrix(features, output: observation)
        let feature_matrix = feature_matrix_and_output.0
        let normalized_feature_matrix = transpose(normalize(feature_matrix))
        let y_predicted = predictEntireMatrixOfFeatures(normalized_feature_matrix, your_weights: self.final_weights)

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
     when fitting your model (using the fit() method). Make sure your first feature is the constant 1 for the intercept.

     - parameter input_vector: An array of input (depends on how many features you used to fit your model)
     - parameter weights: An array of your weights. This can be obtained by fitting your model before getting a prediction.
     - returns: A prediction (of type Float).

     */
    public func predict(input_vector: ValueArray<Float>, your_weights: ValueArray<Float>) -> Float {

        let prediction: Float = input_vector • your_weights

        return prediction
    }

    /**
     The predictEntireMatrixOfFeatures is used for making predictions using all of your feature data. Before using this function normalize your features by calling the MLDataManagers normalizeFeatures static method.

     - parameter input_matrix: You need to utilize the dataToMatrix() method from the MLDataManager class in order to convert your array of features into a matrix. input_matrix is a
     matrix that consists of all of your features.
     - parameter weights: A matrix that consists of your weights.
     - returns: An array of predictions (of type Float)
     */
    public func predictEntireMatrixOfFeatures(input_matrix: Matrix<Float>, your_weights: Matrix<Float>) -> ValueArray<Float> {

        let predictions = input_matrix * your_weights

        return predictions.elements
    }

    /**
     The getCostFunctionResult function returns your cost function result (RSS).
     */
    public func getCostFunctionResult() -> Float {
        return self.cost_function_result
    }

    /**
     The getWeightsAsMatrix function returns your weights.
     */
    public func getWeightsAsMatrix() -> Matrix<Float> {
        return self.final_weights
    }

    /**
     The getWeightsAsValueArray function returns a value array that contains your weights.
     */
    public func getWeightsAsValueArray() -> ValueArray<Float> {
        return self.final_weights.elements
    }
}
