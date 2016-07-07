//
//  Protocols.swift
//  MLKit
//
//  Created by Guled on 7/6/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation

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

