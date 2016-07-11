//
//  RegressionErrorEnum.swift
//  MLKit
//
//  Created by Guled on 7/10/16.
//  Copyright © 2016 Somnibyte. All rights reserved.
//

import Foundation


enum RegressionError:ErrorType{
    case lengthOfDataArrayNotEqual
    case modelHasNotBeenFit
}