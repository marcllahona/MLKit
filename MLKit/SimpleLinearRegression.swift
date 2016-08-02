//
//  SimpleLinearRegression.swift
//  MLKit
//
//  Created by Guled on 6/29/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation
import Upsurge

public class SimpleLinearRegression {

    private var slope: Float!
    private var intercept: Float!
    private var cost_function_result: Float!

    public init() {
        slope = 0.0
        intercept = 0.0
        cost_function_result = 0.0
    }

    /**
     The fitUsingGradientDescent method fits/trains your model (that consists of one feature and one output array) and returns your regression coefficients/weights. The methods applies gradient descent
     as a means to find the most optimal regression coefficients for your model.

     - parameter inputFeature: An array of your feature. Only 1 feature is allowed. This class is used for simple experiments in which only 1 feature is allowed.
     - parameter output: An array of your observations/output.
     - parameter step_size: The amount of "steps" you want to take in the gradient descent process.
     - parameter tolerance: The stopping point. Since it might take awhile to hit 0 you can set a tolerance to stop at a specific point.

     - returns: A tuple of your slope and intercept (your regression coefficients).
     */
    public func train(inputFeature: Array<Float>, output: Array<Float>, step_size: Float, tolerance: Float) throws -> (Float, Float) {

        if (inputFeature.count != output.count) {

            print("The length of your input feature data and your output must be the same in order to utilize this function.")
            throw RegressionError.lengthOfDataArrayNotEqual
        }

        var converged = false // Convergence Boolean
        var current_slope = Float(0.0) // Slope / Weight
        var current_intercept = Float(0.0)
        var predictions_from_training_data = Array<Float>()
        var errors = Array<Float>()
        var sum_of_errors = Float(0.0)
        var adjustment_for_intercept = Float(0.0)
        var error_and_input_sum = Float(0.0)
        var adjustment_for_slope = Float(0.0)
        var gradient_magnitude = Float(0.0)

        // Function to subtract predictions from the actual observations
        let subtract_predictions_from_output = { (predictions: Array<Float>, output: Array<Float>) -> Array<Float> in
            var new_data = Array<Float>()

            for (i, data) in predictions.enumerate() {
                let value = data - output[i]
                new_data.append(value)
            }

            return new_data
        }

        // Function to find the product of our errors and input feature data
        let find_product_of_error_and_input = { () -> Array<Float> in

            var new_data = Array<Float>()

            for (i, data) in errors.enumerate() {
                let value = data * inputFeature[i]
                new_data.append(value)
            }

            return new_data
        }

        while !converged {

            // Compute the predicted values given the current slope and intercept
            predictions_from_training_data = getTrainingDataPredictions(inputFeature, slope: current_slope, intercept: current_intercept)

            // Compute the prediction errors (prediction - Y)
            errors = subtract_predictions_from_output(predictions_from_training_data, output)

            // Update the intercept
            sum_of_errors = sum(errors)
            adjustment_for_intercept = step_size * sum_of_errors
            current_intercept = current_intercept - adjustment_for_intercept

            // Update the slope
            let error_input_arr = find_product_of_error_and_input()
            error_and_input_sum = sum(error_input_arr)
            adjustment_for_slope = step_size * error_and_input_sum
            current_slope = current_slope - adjustment_for_slope

            // Compute the magnitude of the gradient
            let sum_of_errors_squared = (sum_of_errors * sum_of_errors)
            let error_and_input_sum_squared = error_and_input_sum * error_and_input_sum
            gradient_magnitude = sqrt(sum_of_errors_squared + error_and_input_sum_squared)

            // Check for convergence
            if gradient_magnitude < tolerance {
                converged = true
            }

        }

        self.slope = current_slope
        self.intercept = current_intercept

        return (current_slope, current_intercept)
    }

    /**
     The fitUsingNoGradientDescent method fits/trains your model by taking the derivative of the residual sum of squares formula (An .md file with the intuition behind this approach will be provided soon) and solves for your regression coefficients.

     - parameter inputFeature: An array of your feature. Only 1 feature is allowed. This class is used for simple experiments in which only 1 feature is allowed.
     - parameter output: An array of your observations/output.

     - returns: A tuple of your slope and intercept (your regression coefficients).
     */
    public func train(inputFeature: Array<Float>, output: Array<Float>) -> (Float, Float) {

        if (inputFeature.count == 0 || output.count == 0) {
            print("You need to have 1 feature array and 1 output array to utilize this function.")
            return (Float(-1), Float(-1))
        }

        if (inputFeature.count != output.count) {

            print("The length of your input feature data and your output must be the same in order to utilize this function.")
            return (Float(-1), Float(-1))
        }

        // The sum over all of our observations/output (y)
        let y_sub_i = sum(output)

        // The sum over all of our input data (x)
        let x_sub_i = sum(inputFeature)

        // The sum over all of our input data squared
        let data_squared = inputFeature.map { $0 * $0 }
        let x_sub_i_squared = sum(data_squared)

        // The sum over all of our input data and output data
        var input_and_output_multiplied: Array<Float> = []
        for (i, element) in inputFeature.enumerate() {
            let value = element * output[i]
            input_and_output_multiplied.append(value)
        }
        let x_sub_i_y_sub_i = sum(input_and_output_multiplied)

        // Calculate the slope (w1)
        let numerator = x_sub_i_y_sub_i - (Float(1.0) / Float(inputFeature.count)) * (y_sub_i * x_sub_i)
        let denominator = x_sub_i_squared - (Float(1.0) / Float(inputFeature.count)) * (x_sub_i * x_sub_i)
        let slope = numerator / denominator

        // Calculate the intercept
        let intercept = mean(output) - slope * mean(inputFeature)

        // Save the current slope and intercept
        self.slope = slope
        self.intercept = intercept

        return (slope, intercept)
    }

    /**
     The RSS method computes the residual sum of squares or the cost function of your model.

     - parameter inputFeature: An array of your feature. Only 1 feature is allowed. This class is used for simple experiments in which only 1 feature is allowed.
     - parameter output: An array of your observations/output.
     - parameter intercept: Your intercept weight (of type Float).
     - parameter slope: your slope weight (of type Float).

     - returns: The cost of your model (a.k.a The Residual Sum of Squares).

     */
    public func RSS (inputFeature: Array<Float>, output: Array<Float>, slope: Float, intercept: Float) -> Float {
        var sum = Float(0.0)
        for (i, _) in inputFeature.enumerate() {
            let value = (output[i] - predict(slope, intercept: intercept, inputValue: inputFeature[i]))
            let value_squared = value * value
            sum = sum + value_squared
        }

        // Save RSS/Cost
        self.cost_function_result = sum

        return sum
    }

    /**
     The predict method is used for making one-time predictions by passing your intercept weight, slope weight, and an input value.

     - parameter intercept: Your intercept weight (of type Float).
     - parameter slope: your slope weight (of type Float).
     - paramter inputValue: An input value.

     - returns: A prediction (of type Float).
     */
    public func predict (slope: Float, intercept: Float,  inputValue: Float) -> Float {
        let y_hat = intercept + slope * inputValue
        return y_hat
    }

    func getTrainingDataPredictions (inputFeature: Array<Float>, slope: Float, intercept: Float) -> Array<Float> {

        var predictions_from_training_data = Array<Float>()

        for data in inputFeature {
            let yhat = predict(slope, intercept: intercept, inputValue: data)
            predictions_from_training_data.append(yhat)
        }

        return predictions_from_training_data
    }
    
    
    /** 
     The getRegressionCoefficients function returns your slope and intercept.
    */
    public func getRegressionCoefficients() -> (Float,Float) {
        return (self.slope, self.intercept)
    }
    
    /** 
        The getCostFunctionResult function returns your cost function result (RSS).
    */
    public func getCostFunctionResult() -> Float{
        return self.cost_function_result
    }

    
}

