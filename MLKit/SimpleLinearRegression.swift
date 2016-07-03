//
//  SimpleLinearRegression.swift
//  MLKit
//
//  Created by Guled on 6/29/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation


public class SimpleLinearRegression{
    
    
    public var slope:Float!
    public var intercept:Float!
    public var cost_function_result:Float!
    
    public init(){
        slope = 0.0
        intercept = 0.0
        cost_function_result = 0.0
    }
    
    public func fitUsingGradientDescent (inputFeature:Array<Float>, output:Array<Float>, step_size: Float, tolerance: Float) -> (Float, Float){
        
        
        if(inputFeature.count == 0 || output.count == 0){
            print("You need to have 1 feature array and 1 output array to utilize this function.")
            return (Float(-1),Float(-1))
        }
        
        if(inputFeature.count != output.count){
            
            print("The length of your input feature data and your output must be the same in order to utilize this function.")
            return (Float(-1), Float(-1))
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
        let subtract_predictions_from_output = { (predictions:Array<Float>, output:Array<Float>) -> Array<Float> in
            var new_data = Array<Float>()
            
            for(i, data) in predictions.enumerate(){
                let value = data - output[i]
                new_data.append(value)
            }
            
            return new_data
        }
        
        // Function to find the product of our errors and input feature data
        let find_product_of_error_and_input = { () -> Array<Float> in
            
            var new_data = Array<Float>()
            
            for(i, data) in errors.enumerate(){
                let value = data * inputFeature[i]
                new_data.append(value)
            }
            
            return new_data
        }
    
        while !converged {
            
            // Compute the predicted values given the current slope and intercept
            predictions_from_training_data = getTrainingDataPredictions(inputFeature, intercept: current_intercept, slope: current_slope)
            
            // Compute the prediction errors (prediction - Y)
            errors = subtract_predictions_from_output(predictions_from_training_data, output)
            
            // Update the intercept
            sum_of_errors = MLDataManager.sumUpData(errors)
            adjustment_for_intercept = step_size * sum_of_errors
            current_intercept = current_intercept - adjustment_for_intercept
            
            // Update the slope
            let error_input_arr = find_product_of_error_and_input()
            error_and_input_sum = MLDataManager.sumUpData(error_input_arr)
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
    
    
    public func fitUsingNoGradientDescent (inputFeature:Array<Float>, output:Array<Float> ) -> (Float, Float){
        
        
        if(inputFeature.count == 0 || output.count == 0){
            print("You need to have 1 feature array and 1 output array to utilize this function.")
            return (Float(-1),Float(-1))
        }
        
        if(inputFeature.count != output.count){
            
            print("The length of your input feature data and your output must be the same in order to utilize this function.")
            return (Float(-1), Float(-1))
        }
        
        // The sum over all of our observations/output (y)
        let y_sub_i = MLDataManager.sumUpData(output)
        
        // The sum over all of our input data (x)
        let x_sub_i = MLDataManager.sumUpData(inputFeature)
        
        // The sum over all of our input data squared
        let data_squared = inputFeature.map { $0 * $0}
        let x_sub_i_squared = MLDataManager.sumUpData(data_squared)
        
        // The sum over all of our input data and output data
        var input_and_output_multiplied:Array<Float> = []
        for (i, element) in inputFeature.enumerate() {
            let value = element * output[i]
            input_and_output_multiplied.append(value)
        }
        let x_sub_i_y_sub_i = MLDataManager.sumUpData(input_and_output_multiplied)
        
        // Calculate the slope (w1)
        let numerator =  x_sub_i_y_sub_i - (Float(1.0) / Float(inputFeature.count)) * (y_sub_i * x_sub_i)
        let denominator = x_sub_i_squared  - (Float(1.0) / Float(inputFeature.count)) * (x_sub_i * x_sub_i)
        let slope = numerator/denominator
        
        // Calculate the intercept 
        let intercept = MLDataManager.mean(output) - slope * MLDataManager.mean(inputFeature)
        
        
        // Save the current slope and intercept 
        self.slope = slope
        self.intercept = intercept
        
        return (slope, intercept)
    }
    
    public func RSS (inputFeature: Array<Float>, output:Array<Float>, intercept:Float, slope: Float) -> Float {
        var sum = Float(0.0)
        for (i, _) in inputFeature.enumerate() {
           let value = (output[i] - predict(intercept, slope: slope, inputValue: inputFeature[i]))
           let value_squared = value * value
           sum = sum + value_squared
        }
    
        // Save RSS/Cost 
        self.cost_function_result = sum
        
        return sum
    }
    
    public func predict (intercept: Float, slope: Float, inputValue: Float) -> Float {
        let y_hat = intercept + slope * inputValue
        return y_hat
    }
    
    func getTrainingDataPredictions (inputFeature: Array<Float>, intercept: Float, slope: Float) -> Array<Float> {
        
        var predictions_from_training_data = Array<Float>()
        
        for data in inputFeature{
            let yhat = predict(intercept, slope: slope, inputValue: data)
            predictions_from_training_data.append(yhat)
        }

        return predictions_from_training_data
    }
}




















