//
//  DataManager.swift
//  MLKit
//
//  Created by Guled on 6/30/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation
import Upsurge

public class MLDataManager {

    enum MLDataHandelingError: ErrorType {
        case noData
        case incorrectFraction
        case unacceptableInput
    }

    /**
     The sumUpData method takes in array of numbers (of any type using the NumericType protocol) and computes the sum of all the numbers in the array.

     - parameter data: An array of numbers.

     - returns: The sum of the array
     */
    public static func sumUpData<T: NumericType> (data: Array<T>) -> T {
        var sum = T(0)
        for val in data {
            sum = sum + val
        }
        return sum
    }

    /**
     The mean method calculates the mean of an array of numbers (of any type using the NumericType protocol).

     - parameter data: An array of numbers.

     - returns: The mean of the array
     */
    public static func mean<T: NumericType> (data: Array<T>) -> T {
        let totalSum = sumUpData(data)
        let totalAmountOfData = T(data.count)
        return totalSum / totalAmountOfData
    }

    /**
     The dataToMatrix method takes an array of features (which contain your data of a specific feature), along with your observations/output
     and turns your features into a Matrix of type Float and your output into an array in order to be processed by machine learning algorithms
     such as calculating the RSS/cost function for Regression.

     - parameter features: An array of your features (which are suppose to be arrays as well).
     - parameter output: An array of your observations/output.

     - returns: A tuple that consists of a matrix and a array of type ValueArray
     */
    public static func dataToMatrix (features: [Array<Float>], output: Array<Float>) -> (Matrix<Float>, ValueArray<Float>) {

        // Create Output Matrix
        let output_matrix = Matrix<Float>(rows: output.count, columns: 1, elements: output)

        // Create "contant/intercept" list
        let contant_array = [Float](count: features[0].count, repeatedValue: 1.0)
        var matrix_as_array: [[Float]] = []

        for (i, _) in contant_array.enumerate() {
            var new_row: [Float] = []
            new_row.append(contant_array[i])

            for feature_array in features {
                new_row.append(feature_array[i])
            }

            matrix_as_array.append(new_row)
        }

        let feature_matrix = Matrix<Float>(matrix_as_array)

        return (feature_matrix, output_matrix.elements)
    }

    /**
     A method that takes in a string of arrays (if you read your data in from a CSV file) and converts it into an array of Float values.

     - parameter data: A string array that contains your feature data.

     - returns: An array of type Float.
     */
    public static func convertMyDataToFloat(data: Array<String>) throws -> Array<Float> {

        if data.count == 0 {
            throw MLDataHandelingError.noData
        }

        let float_data: Array<Float> = data.map { Float($0)! }
        return float_data
    }

    /**
     The split data method allows you to split your original data into training and testing sets (or training,validation, and testing sets). The method takes in your data
     and a fraction and splits your data based on the fraction you specify. So for example if you chose 0.5 (50%), you would get a tuple containing two halves of your data.

     - parameter data: An array of your feature data.
     - parameter fraction: The amount you want to split the data.

     - returns: A tuple that contains your split data. The first entry of your tuple (0) will contain the fraction of data you specified, and the last entry of your tuple (1) will
     contain whatever data is left.
     */
    public static func splitData(data: Array<Float>, fraction: Float) throws -> (Array<Float>, Array<Float>) {

        if data.count == 0 {
            throw MLDataHandelingError.noData
        }

        if (fraction == 1.0 || fraction == 0.0 || fraction >= 1.0) {
            print("Your fraction must be between 1.0 and 0.0")
            throw MLDataHandelingError.incorrectFraction
        }

        let dataCount = Float(data.count)
        let split = Int(fraction * dataCount)
        let firstPortion = data[0 ..< split]
        let secondPortion = data[split ..< data.count]
        let firstPortionAsArray = Array(firstPortion)
        let secondPortionAsArray = Array(secondPortion)

        return (firstPortionAsArray, secondPortionAsArray)
    }

    /**
     The randomlySplitData method allows you to split your original data into training and testing sets (or training,validation, and testing sets). The method takes in your data
     (shuffles it in order to make it completely random) and a fraction and splits your data based on the fraction you specify. So for example if you chose 0.5 (50%), you would get a tuple containing two halves of your data (that have been randomly shuffled).

     - parameter data: An array of your feature data.
     - parameter fraction: The amount you want to split the data.

     - returns: A tuple that contains your split data. The first entry of your tuple (0) will contain the fraction of data you specified, and the last entry of your tuple (1) will
     contain whatever data is left.
     */
    public static func randomlySplitData(data: Array<Float>, fraction: Float) throws -> (Array<Float>, Array<Float>) {

        if data.count == 0 {
            throw MLDataHandelingError.noData
        }

        if (fraction == 1.0 || fraction == 0.0 || fraction >= 1.0) {
            print("Your fraction must be between 1.0 and 0.0")
            throw MLDataHandelingError.incorrectFraction
        }

        // Shuffle the users input
        var shuffledData = data.shuffle()

        let dataCount = Float(data.count)
        let split = Int(fraction * dataCount)
        let firstPortion = shuffledData[0 ..< split]
        let secondPortion = shuffledData[split ..< data.count]
        let firstPortionAsArray = Array(firstPortion)
        let secondPortionAsArray = Array(secondPortion)

        return (firstPortionAsArray, secondPortionAsArray)
    }

    /**
     The convertDataToPolynomialOfDegree function takes your array of data from 1 feature and allows you to create complex models
     by raising your data up to the power of the degree parameter. For example if you pass in 1 feature (ex: [1,2,3]), and a degree of 3, the method
     will return 3 arrays as follows:  [ [1,2,3], [1,4,9], [1, 8, 27] ].

     - parameter data: An array of your feature data.

     - returns: An array of your features. The first will be your original data that was passed in since all the data within your feature was already raised to the
     first power. Subsequent arrays will consist of your data being raised up to a certain degree.
     */
    public static func convertDataToPolynomialOfDegree(data: Array<Float>, degree: Int) throws -> [Array<Float>] {

        if data.count == 0 {
            throw MLDataHandelingError.noData
        }

        if degree < 1 {
            print("Degree must be greater than 1.")
            throw MLDataHandelingError.unacceptableInput
        }

        // Array of features
        var features: [Array<Float>] = []

        // Set the feature passed in as the first entry of the array since this is considered as "power_1" or "to the power of 1"
        features.append(data)

        if degree > 1 {
            // Loop over remaining degrees
            // range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
            for power in 2..<(degree + 1) {
                let new_feature = data.map { powf($0, Float(power)) }
                features.append(new_feature)
            }
        }

        return features
    }

}

