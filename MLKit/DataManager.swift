//
//  DataManager.swift
//  MLKit
//
//  Created by Guled on 6/30/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation
import Upsurge

/**

 The Numeric Type Protocol allows for generic functions
 to take in integers or doubles or floats.

 */
public protocol NumericType {
    func + (lhs: Self, rhs: Self) -> Self
    func - (lhs: Self, rhs: Self) -> Self
    func * (lhs: Self, rhs: Self) -> Self
    func / (lhs: Self, rhs: Self) -> Self
    func % (lhs: Self, rhs: Self) -> Self
    init(_ v: Int)
}

extension Double: NumericType { }
extension Float: NumericType { }
extension Int: NumericType { }
extension Int8: NumericType { }
extension Int16: NumericType { }
extension Int32: NumericType { }
extension Int64: NumericType { }
extension UInt: NumericType { }
extension UInt8: NumericType { }
extension UInt16: NumericType { }
extension UInt32: NumericType { }
extension UInt64: NumericType { }

public class MLDataManager {

    enum MLDataHandelingError: ErrorType {
        case noData
        case incorrectFraction
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

     - parameter data: A string array.

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

     - parameter data: An array of your float data
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

}

