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
    func +(lhs: Self, rhs: Self) -> Self
    func -(lhs: Self, rhs: Self) -> Self
    func *(lhs: Self, rhs: Self) -> Self
    func /(lhs: Self, rhs: Self) -> Self
    func %(lhs: Self, rhs: Self) -> Self
    init(_ v: Int)
}


extension Double : NumericType { }
extension Float  : NumericType { }
extension Int    : NumericType { }
extension Int8   : NumericType { }
extension Int16  : NumericType { }
extension Int32  : NumericType { }
extension Int64  : NumericType { }
extension UInt   : NumericType { }
extension UInt8  : NumericType { }
extension UInt16 : NumericType { }
extension UInt32 : NumericType { }
extension UInt64 : NumericType { }


public class MLDataManager{

    
    enum MLDataHandelingError: ErrorType {
        
    }
    
    
    /**
     Description Goes Here
     
     @param
     
     @return 
     
     */
    public static func sumUpData<T: NumericType> (data: Array<T>) -> T {
        var sum = T(0)
        for val  in data{
            sum = sum + val
        }
        return sum
    }
    
    /**
     Description Goes Here
     
     @param
     
     @return
     
     */
    public static func mean<T: NumericType> (data: Array<T>) -> T {
        let totalSum = sumUpData(data)
        let totalAmountOfData = T(data.count)
        return totalSum/totalAmountOfData
    }
    
    public static func dataToMatrix (features:[Array<Float>], output: Array<Float>) -> (Matrix<Float>,ValueArray<Float>) {
        
        
        // Create Output Matrix 
        let output_matrix = Matrix<Float>(rows:output.count, columns: 1, elements:output)
        
        // Create "contant/intercept" list 
        let contant_array = [Float](count: features[0].count, repeatedValue: 1.0)
        var matrix_as_array:[[Float]] = []
        
        for (i, _) in contant_array.enumerate() {
            var new_row:[Float] = []
            new_row.append(contant_array[i])
            
            for feature_array in features {
                new_row.append(feature_array[i])
            }
            
            matrix_as_array.append(new_row)
        }
        
        let feature_matrix = Matrix<Float>(matrix_as_array)
        
        return (feature_matrix, output_matrix.elements)
    }
    

    
}















